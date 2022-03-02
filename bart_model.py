import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import transformers
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import AdamW, Adafactor
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    MBartForConditionalGeneration,
)

from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.seq2seq.seq2seq_utils import SimpleSummarizationDataset

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

if transformers.__version__ < "4.2.0":
    MBartForConditionalGeneration._keys_to_ignore_on_save = []

logger = logging.getLogger(__name__)


class BartModel:
    def __init__(
        self,
        pretrained_model=None,
        additional_special_tokens_encoder=None,
        additional_special_tokens_decoder=None,
        model_config=None,
        vocab_file=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):
        self.args = self._load_model_args()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        # BartConfig, BartForConditionalGeneration, BartTokenizer
        # config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)

        if pretrained_model is None:
            model_config = BartConfig.from_json_file(model_config)
            self.model = BartForConditionalGeneration(config=model_config)
            self.encoder_tokenizer = BartTokenizer.from_pretrained(vocab_file)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(pretrained_model)
            self.encoder_tokenizer = BartTokenizer.from_pretrained(pretrained_model)
        self.decoder_tokenizer = self.encoder_tokenizer
        self.model.resize_token_embeddings(len(self.encoder_tokenizer))
        self.config = self.model.config

        if additional_special_tokens_encoder is not None:
            self.encoder_tokenizer.add_special_tokens(additional_special_tokens_encoder)

        if additional_special_tokens_decoder is not None:
            self.decoder_tokenizer.add_special_tokens(additional_special_tokens_decoder)

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

        self.args.model_type = 'bart'
        self.args.model_name = 'ComFormer'

    def train_model(
        self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        # if self.args.silent:
        #     show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(self.args.output_dir, model=self.model)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

        return global_step, training_details

    def train(
        self, train_dataset, output_dir, show_running_loss=True, eval_data=None, verbose=True, **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        # TODO: Use custom optimizer like with BertSum?
        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt")))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info(" Training started")

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for current_epoch in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self.save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
                    self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True, verbose=verbose, silent=silent)
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
        self.results.update(result)

        if self.args.evaluate_generated_text:
            to_predict = eval_data["input_text"].tolist()
            preds = self.predict(to_predict)

            result = self.compute_metrics(
                eval_data["target_text"].tolist(), preds, **kwargs
            )
            self.results.update(result)

        if verbose:
            logger.info(self.results)

        return self.results

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
            # batch = tuple(t.to(device) for t in batch)

            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss = outputs[0]
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        results["eval_loss"] = eval_loss

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()

        all_outputs = []
        # Batching
        for batch in tqdm(
            [
                to_predict[i : i + self.args.eval_batch_size]
                for i in range(0, len(to_predict), self.args.eval_batch_size)
            ],
            desc="Generating outputs",
            disable=self.args.silent,
        ):
            input_ids = self.encoder_tokenizer.batch_encode_plus(
                batch,
                max_length=self.args.max_seq_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )["input_ids"].to(self.device)

            if self.args.model_type in ["bart", "marian"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )

            all_outputs.extend(outputs.cpu().numpy())

        outputs = [
            self.decoder_tokenizer.decode(
                output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True
            )
            for output_id in all_outputs
        ]

        if self.args.num_return_sequences > 1:
            return [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(
            output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True
        )

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """

        results = {}
        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.model_type in ["bart", "mbart", "marian"]:
            dataset = SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)
            return dataset

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.no_save:
            model_to_save = model.module if hasattr(model, "module") else model
            self.save_model_args(output_dir)

            if self.args.model_type in ["bart", "mbart", "marian"]:
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                model_to_save.save_pretrained(output_dir)
                self.config.save_pretrained(output_dir)
                if self.args.model_type in ["bart", "mbart"]:
                    self.encoder_tokenizer.save_pretrained(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        device = self.device
        if self.args.model_type in ["bart", "marian"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = (
                batch["source_ids"],
                batch["source_mask"],
                batch["target_ids"],
            )
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }

        return inputs

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir=None):
        args = Seq2SeqArgs()
        if input_dir is not None:
            args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
