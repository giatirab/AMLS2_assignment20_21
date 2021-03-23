from torchtext.legacy import data
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import gzip
import shutil
from tqdm import tqdm
import os
from argparse import ArgumentParser
import json
import spacy
from collections import (
    Counter,
)  # oggetto che data una lista calcola le frequenze degli oggetti in essa contenuti
import itertools
import numpy as np
import random
import dill
import hashlib
from modules import Transformer


class TrasformerManager:
    def __init__(self):
        self.tokenize = spacy.load("en_core_web_sm").tokenizer

        self.label_field = data.Field()
        self.text_field = data.Field(
            tokenize=lambda txt: txt.split(), batch_first=True
        )
        if not (
            os.path.exists("models/label_field.pt")
            and os.path.exists("models/text_field.pt")
        ):
            self.preprocess()
        else:
            with open("models/text_field.pt", "rb") as f:
                self.text_field = dill.load(f)
            with open("models/label_field.pt", "rb") as f:
                self.label_field = dill.load(f)

        parser = ArgumentParser()
        with open("data/parameters.json") as f:
            default_args = json.load(f)
        for key, val in default_args.items():
            parser.add_argument(
                f'--{key.replace("_","-")}',
                dest=key,
                default=val,
                type=type(val),
            )
        args = parser.parse_args()
        models_path = "models"
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        self.vocab = set(self.text_field.vocab.itos)
        args_dict = {key: args.__dict__[key] for key in default_args}
        args_json = json.dumps(args_dict, indent=4, sort_keys=True)
        print(args_json)
        args_dict["model_name"] = (
            args_dict["model_name"]
            if args_dict["model_name"] != ""
            else os.path.join(
                models_path, hashlib.md5(args_json.encode()).hexdigest()
            )
        )
        self.__dict__.update(args_dict)

        self.avg_rec = (0, 0.0)
        self.writer = SummaryWriter()

        self.num_labels = len(self.label_field.vocab) - 2
        print(f"Num labels: {self.num_labels}")
        print(f"Vocab size: {len(self.vocab)}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # embedding_dim = 128
        self.model = Transformer(
            embedding_dim=128,
            seq_length=self.max_tweet_len,
            num_heads=self.heads,
            num_tokens=len(self.text_field.vocab),
            depth=self.depth,
            num_labels=self.num_labels,
            output_dropout=self.output_dropout,
            block_dropout=self.block_dropout,
        ).to(self.device)
        if os.path.exists(self.model_name):
            self.model.load_state_dict(
                torch.load(self.model_name, map_location=self.device)
            )
            print("Loaded pretrained model:", self.model_name)
        else:
            print("Model:", self.model_name, "doesn't exists.")

    def train(self):
        data_path = "data"
        data_file_names = ["train_dataset.csv.gz", "test_dataset.csv.gz"]
        for i, data_file_name in enumerate(data_file_names):
            data_file_name = os.path.join(data_path, data_file_name)
            uncompressed_data_file_name = ".".join(
                data_file_name.split(".")[:-1]
            )
            if data_file_name.split(".")[-1] == "gz" and not os.path.exists(
                uncompressed_data_file_name
            ):
                print("Uncompressing data")
                with gzip.open(data_file_name, "rb") as f_in:
                    with open(uncompressed_data_file_name, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            data_file_names[i] = uncompressed_data_file_name.split("/")[1]
        print("Extracting datasets")
        train_val_dataset, test_dataset = data.TabularDataset.splits(
            data_path,
            train="train_dataset.csv",
            test="test_dataset.csv",
            fields=(("label", self.label_field), ("tweet", self.text_field)),
            format="csv",
            skip_header=True,
        )
        train_dataset, val_dataset = train_val_dataset.split(0.9)
        optimizer = torch.optim.Adam(
            lr=self.lr, params=self.model.parameters()
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda i: min(i / (self.lr_warmup / self.batch_size), 1.0),
        )

        print("Creating batch iterators")
        val_data_iter = data.BucketIterator(
            val_dataset,
            batch_size=self.test_batch_size,
            device=self.device,
            shuffle=True,
        )
        train_data_iter = data.BucketIterator(
            train_dataset,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=True,
        )
        test_data_iter = data.BucketIterator(
            test_dataset,
            batch_size=self.test_batch_size,
            device=self.device,
            shuffle=True,
        )
        avg_val_loss, avg_val_rec = self._test(0, val_data_iter)
        print(
            f"AvgRec: {round(avg_val_rec, 4)},\tavg loss: {round(avg_val_loss, 6)},\tepoch: 0\n"
        )
        log_count = 0
        step_loss = 0
        tot_loss = 0
        for epoch in range(self.epochs):
            self.model.train()
            for batch in tqdm(train_data_iter):
                optimizer.zero_grad()
                label = batch.label - 2
                output = self.model(batch.tweet)
                loss = F.nll_loss(output, label[0])
                step_loss += loss.item()
                loss.backward()
                if self.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                optimizer.step()
                scheduler.step()
                log_count += len(batch)
                if log_count >= self.log_step:
                    step_avg_loss = step_loss / log_count
                    tot_loss += step_loss
                    print(
                        f"Train loss: {round(step_avg_loss, 6)}\tEpoch: {epoch}"
                    )
                    log_count = 0
                    step_loss = 0.0

            torch.save(self.model.state_dict(), self.model_name)
            avg_train_loss = tot_loss / len(train_dataset)
            self.writer.add_scalar(
                "train_avg_loss", avg_train_loss, global_step=epoch
            )
            print(f"Train avg loss: {round(avg_train_loss, 6)}")
            avg_val_loss, avg_val_rec = self._test(epoch + 1, val_data_iter)
            self.writer.add_scalar(
                "avg_val_rec", avg_val_rec, global_step=epoch
            )
            self.writer.add_scalar(
                "avg_val_loss", avg_val_loss, global_step=epoch
            )
            print(
                f"AvgRec: {round(avg_val_rec, 4)},\tavg loss: {round(avg_val_loss, 6)},\tepoch: {epoch}\n"
            )
            tot_loss = 0
            step_avg_loss = 0
            log_count = 0
        print("Performing final test")
        test_avg_loss, test_avg_rec = self._test(-1, test_data_iter)
        test_avg_loss = round(test_avg_loss, 6)
        self.writer.add_text("results", f"test_avg_loss {test_avg_loss}")
        self.writer.add_text("results", f"test_avg_rec {test_avg_rec}")
        print(
            f"AvgRec: {round(test_avg_rec, 4)},\tavg loss: {test_avg_loss},\tepoch: {epoch}\n"
        )
        print(
            f"Best val AvgRec:{round(self.avg_rec[1], 3)} at epoch:{self.avg_rec[0]}"
        )

    def _test(self, epoch, data_iter):
        with torch.no_grad():
            self.model.eval()
            tot_loss = 0.0
            correct_count = torch.zeros(self.num_labels, 2)
            dataset_size = 0
            for batch in data_iter:
                labels = batch.label[0] - 2
                output = self.model(batch.tweet)
                predicted_labels = output.max(dim=1).indices.cpu()
                for i, label in enumerate(labels.cpu()):
                    correct_count[label][0] += predicted_labels[i] == label
                    correct_count[label][1] += 1
                loss = F.nll_loss(output, labels)
                tot_loss += loss.item()
                dataset_size += len(batch)
            avg_rec = (
                sum(
                    [
                        correct / count if count > 0 else 0
                        for correct, count in correct_count
                    ]
                )
                / self.num_labels
            ).item()
            if avg_rec > self.avg_rec[1]:
                self.avg_rec = (epoch, avg_rec)
            test_loss = tot_loss / dataset_size
        return test_loss, avg_rec

    def classify(self, tweet):
        # ipdb.set_trace()
        with torch.no_grad():
            self.model.eval()
            full_tokens = [
                token.text.lower() for token in self.tokenize(tweet)
            ]
            tokens = [token for token in full_tokens if token in self.vocab]
            not_known = [token for token in full_tokens if token not in tokens]
            if len(not_known) > 0:
                print(f"Unknown words: {not_known}")
            token_indexes = [self.text_field.vocab[token] for token in tokens]
            token_indexes = torch.unsqueeze(
                torch.tensor(token_indexes, dtype=torch.long), 0
            )
            output = self.model(token_indexes.to(self.device))[0]
            predicted_label_index = output.max(dim=0).indices.item()
            index_to_label = {
                val: key for key, val in self.label_field.vocab.stoi.items()
            }
            print(
                f"Confidence: {round(100*torch.exp(output[predicted_label_index]).item(),1)}%"
            )
            return (
                "positive"
                if index_to_label[predicted_label_index + 2] == "4"
                else "negative"
            )

    def preprocess(self):
        print("Preprocessing started")

        df = pd.read_csv(
            "data/training.1600000.processed.noemoticon.csv.gz",
            encoding="ISO-8859-1",
        )
        df.columns = ["label", "id", "date", "unknown", "author", "tweet"]
        df = df.drop(columns=["date", "unknown", "author", "id"])
        df = df.reindex(columns=["label", "tweet"])
        indexes = list(range(len(df)))
        random.shuffle(indexes)
        split_index = int(np.floor(len(df) / 10))
        test_df = df.loc[indexes[:split_index]]
        train_df = df.loc[indexes[split_index:]]
        train_tweets_tokens = tuple(
            tuple(token.text.lower() for token in self.tokenize(tweet))
            for tweet in tqdm(train_df["tweet"])
        )
        test_tweets_tokens = tuple(
            tuple(token.text.lower() for token in self.tokenize(tweet))
            for tweet in tqdm(test_df["tweet"])
        )

        print("Tokenization complete")
        counter = Counter(itertools.chain(*train_tweets_tokens))
        common_words = [item[0] for item in counter.most_common(20000)]
        with open("data/words.json") as f:
            json.dump(common_words, f)
        common_words = set(common_words)
        simplified_train_tweets = tuple(
            " ".join(tuple(token for token in tokens if token in common_words))
            for tokens in tqdm(train_tweets_tokens)
        )
        simplified_test_tweets = tuple(
            " ".join(tuple(token for token in tokens if token in common_words))
            for tokens in tqdm(test_tweets_tokens)
        )
        print("Extracted simplified tweets, saving...")
        test_df = test_df.assign(tweet=simplified_test_tweets)
        train_df = train_df.assign(tweet=simplified_train_tweets)
        train_df.to_csv("data/train_dataset.csv", index=False)
        test_df.to_csv("data/test_dataset.csv", index=False)
        print("Generating vocabulary")
        train_dataset, test_dataset = data.TabularDataset.splits(
            "data",
            train="train_dataset.csv",
            test="test_dataset.csv",
            fields=(("label", self.label_field), ("tweet", self.text_field)),
            format="csv",
            skip_header=True,
        )
        self.label_field.build_vocab(train_dataset)
        self.text_field.build_vocab(train_dataset)
        print("Computing max len")
        max_tweet_len = 0
        for name in (train_dataset, test_dataset):
            for i in range(len(name)):
                length = len(name[i].tweet)
                if length > max_tweet_len:
                    max_tweet_len = length

        with open("data/parameters.json", "r") as f:
            parameters = json.load(f)
        parameters["max_tweet_len"] = max_tweet_len
        print("Saving fields and parameters")
        with open("data/parameters.json", "w") as f:
            json.dump(parameters, f)

        with open("models/label_field.pt", "wb") as f:
            dill.dump(self.label_field, f)
        with open("models/text_field.pt", "wb") as f:
            dill.dump(self.text_field, f)

        print("Compressing CSVs")
        for name in ("train", "test"):
            with open(f"data/{name}_dataset.csv", "rb") as f_in:
                with gzip.open(f"data/{name}_dataset.csv.gz", "wb") as f_out:
                    f_out.writelines(f_in)
            os.remove(f"data/{name}_dataset.csv")
        print("Done preprocessing")


def main():
    tm = TrasformerManager()
    # tm.preprocess()
    # tm.train()
    tweets = (
        "thought sleeping in was an option tomorrow but realizing that it now is not. evaluations in the morning and work in the afternoon!",
        "I hate everything and the world sucks",
        "I love you and the world is beautiful",
        "I fell in love with you",
        "Do you want to merry me?",
        "I'll kick your ass",
        "This water is tasty",
        "This food is amazing",
        "When I'm with you I feel like I'm complete.",
        "Studying all day makes me deeply satisfied",
        "I can't stand you any more, we better not see each others again.",
        "I think I like you",
    )
    print()
    for tweet in tweets:
        print(tweet)
        print(tm.classify(tweet), '\n')


if __name__ == "__main__":
    main()
