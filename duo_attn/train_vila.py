import sys
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import wandb
import matplotlib.pyplot as plt
import os.path as osp

import re
from io import BytesIO
import json
import time

import requests
import torch
from PIL import Image

from duo_attn.utils import (
    get_model,
    parse_args,
    get_tokenizer,
    visualize_pruned_attention_heads,
    full_attention_heads_to_list,
    save_full_attention_heads,
    seed_everything,
)
from duo_attn.data import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader,
)
from duo_attn.patch import (
    enable_duo_attention_training,
    get_full_attention_heads,
    set_full_attention_heads,
    map_full_attention_heads,
    load_full_attention_heads,
)

from duo_attn.loss import l1_loss


import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
import types

from transformers import AutoModelForCausalLM, AutoConfig

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)

def get_prompt(question, options):
  prompt = f"<video>\n {question} Answer with just a single letter corresponding to the option."
  option_letters = 'ABCD'
  for i, option in enumerate(options):
    prompt += f"\n{option_letters[i]}. {option}"
  return prompt

def train(
    args, model, image_processor, tokenizer, rank, world_size, train_dataloader, optimizer, scheduler, resume_step
):
    model.train()

    if rank == 0:
        pbar = tqdm(range(args.num_steps))

    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    while True:
        if global_step >= args.num_steps:
            break
        for step, item in enumerate(train_dataloader):
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping step {global_step} to resume to {resume_step}"
                    )
                continue

            @torch.no_grad()
            def clamp_(x, min_val, max_val):
                x.clamp_(min_val, max_val)

            map_full_attention_heads(model.llm, func=lambda x: clamp_(x, 0, 1))

            video_file = args.video_dir + item['video'][1:]
            from llava.mm_utils import opencv_extract_frames
            images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)

            qs = get_prompt(item['question'], item['options'])
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if DEFAULT_IMAGE_TOKEN not in qs:
                    # do not repeatively append the prompt.
                    if model.config.mm_use_im_start_end:
                        qs = (image_token_se + "\n") * len(images) + qs
                    else:
                        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

            conv_mode = "llava_v0"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_mask = input_ids.new_ones(input_ids.shape)
            outputs = model(
                input_ids,
                images=[
                    images_tensor,
                ],
                attention_mask=attention_mask
            )

            hidden_states = outputs.logits
            original_hidden_states = hidden_states[:1]
            pruned_hidden_states = hidden_states[1:]

            distill_loss = (
                (original_hidden_states - pruned_hidden_states)
                .pow(2)
                .mean(dim=-1)
                .sum()
                * world_size
            )

            full_attention_heads = get_full_attention_heads(model.llm)
            full_attention_heads = [
                h.data.to(original_hidden_states.device)
                for h in full_attention_heads
            ]

            reg_loss = l1_loss(torch.cat(full_attention_heads).float())

            loss = distill_loss + args.reg_weight * reg_loss

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.AVG)

            if local_step != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if rank == 0:
                full_attention_heads_list = full_attention_heads_to_list(
                    full_attention_heads
                )

                if not args.disable_wandb:
                    fig = visualize_pruned_attention_heads(full_attention_heads_list)

                    sample_len = item["input_ids"].shape[1]
                    wandb.log(
                        {
                            "distill_loss": distill_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "attn_heads": fig,
                            "step": global_step,
                            "sample_len": sample_len,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    plt.close(fig)

                pbar.set_description(
                    f"Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:
                if rank == 0:
                    save_full_attention_heads(
                        full_attention_heads_list,
                        os.path.join(
                            args.output_dir,
                            f"full_attention_heads_step={global_step}.tsv",
                        ),
                    )
                    os.system(f"rm {args.output_dir}/full_attention_heads_latest.tsv")
                    os.system(
                        f"cp {args.output_dir}/full_attention_heads_step={global_step}.tsv {args.output_dir}/full_attention_heads_latest.tsv"
                    )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}-rank={rank}.pt",
                    ),
                )

                # copy the full_attention_heads and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}-rank={rank}.pt {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name_from_path(args.model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_name, model_name, args.model_base)

    enable_duo_attention_training(
      model.llm,
      sink_size=args.sink_size,
      recent_size=args.recent_size,
      max_length=args.context_length_max,
      initial_value=1.0,
      enable_ulysses_attention=False,
      streaming_attn_implementation="sdpa"
    )

    for param in model.parameters():
        param.requires_grad = False

    num_attn_heads = 0
    for name, param in model.named_parameters():
        if "full_attention_heads" in name:
            param.requires_grad = True
            num_attn_heads += param.numel()

    setup()

    torch.cuda.set_device(local_rank)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    apply_activation_checkpointing(model)

    if rank == 0:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"Trainable parameter: {name} with shape {param.shape}, dtype {param.dtype}, device {param.device}"
                )

    with open(args.anno_path, 'r') as file:
        data = json.load(file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / (args.num_steps // 5), 0.1),
            max((args.num_steps - step) / (args.num_steps // 5), 0.1),
        ),
    )
    if rank == 0:
        experiment_config = vars(args)
        if not args.disable_wandb:
            wandb.init(project="DuoAttention", config=experiment_config)
            if args.exp_name is not None:
                wandb.run.name = args.exp_name

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f)

    # if resume and link exists, load the latest state
    if args.resume and os.path.exists(
        os.path.join(
            args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
        )
    ):
        # load the latest state in the output_dir
        state = torch.load(
            os.path.join(
                args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
            )
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        full_attention_heads = load_full_attention_heads(
            args.output_dir, filename="full_attention_heads_latest.tsv"
        )
        set_full_attention_heads(model.llm, full_attention_heads)
        resume_step = state["global_step"]
        print(f"Resuming from step {resume_step}")
    else:
        resume_step = -1

    train(
        args,
        model,
        image_processor,
        tokenizer,
        rank,
        world_size,
        data,
        optimizer,
        scheduler,
        resume_step,
    )

    full_attention_heads = get_full_attention_heads(model.llm)
    full_attention_heads = [h.data for h in full_attention_heads]

    if rank == 0:
        print("Training finished")
        if args.output_dir is not None:
            full_attention_heads_list = full_attention_heads_to_list(
                full_attention_heads
            )
            # save the full attention heads as tsv
            save_full_attention_heads(
                full_attention_heads_list,
                os.path.join(args.output_dir, "full_attention_heads.tsv"),
            )

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
