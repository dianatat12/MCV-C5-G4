import torch
from torch import nn, optim
import torchvision.models as models
import transformers
import pytorch_lightning as pl
import numpy as np


class ResNet50(nn.Module):
    def __init__(self, finetune=False):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        self.finetune = finetune

    def forward(self, x):

        if self.finetune:
            features = self.resnet50(x)
        else:
            with torch.no_grad():
                features = self.resnet50(x)

        feature_vector = features.squeeze(0).squeeze(1).squeeze(1)
        return feature_vector


class BERT_Module(nn.Module):
    def __init__(self, finetune=False, version="base"):
        super(BERT_Module, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(f"bert-{version}-uncased")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            f"bert-{version}-uncased"
        )
        self.finetune = finetune

    def forward(self, text):
        if self.finetune:
            tokens = self.tokenizer(text, return_tensors="pt")
            outputs = self.bert(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            with torch.no_grad():
                tokens = self.tokenizer(text, return_tensors="pt")
                outputs = self.bert(**tokens)
                embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings.squeeze(0)
        return embeddings


class TextReprojection(nn.Module):
    def __init__(self, input_size, output_size):
        super(TextReprojection, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class Word2Vec_Module(nn.Module):
    def __init__(self, finetune=False):
        super(Word2Vec_Module, self).__init__()
        self.finetune = finetune

    def forward(self, text):
        pass


class TripletNetwork(pl.LightningModule):
    def __init__(
        self,
        loss=None,
        optimizer=None,
        lr_scheduler=None,
        mode="img_retrieval",
        image_encoder=None,
        text_encoder=None,
        text_reprojection=None,
        from_embeddings=False,
    ):
        super().__init__()

        self.criterion = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.from_embeddings = from_embeddings

        self.text_reprojection = text_reprojection

        self.training_losses = []
        self.validation_losses = []

    def get_image_encoder_output_size(self):
        random_input = torch.randn(1, 3, 224, 224)
        resnet_output = self.image_encoder(random_input)

        return resnet_output.shape[0]

    def get_text_encoder_output_size(self):
        input_text = "This is a sample sentence."
        bert_output = self.text_encoder(input_text)

        return bert_output.shape[0]

    def training_step(self, batch, batch_idx):
        self.model.train()

        anchors, positives, negatives = batch
        if not self.from_embeddings:
            if self.mode == "img_retrieval":
                anchors = self.text_encoder(anchors)
                positives = self.image_encoder(positives)
                negatives = self.image_encoder(negatives)
            elif self.mode == "text_retrieval":
                anchors = self.image_encoder(anchors)
                positives = self.text_encoder(positives)
                negatives = self.text_encoder(negatives)

        if self.mode == "img_retrieval":
            anchors = self.text_reprojection(anchors)

        elif self.mode == "text_retrieval":
            positives = self.text_reprojection(positives)
            negatives = self.text_reprojection(negatives)

        loss = self.criterion(anchors, positives, negatives)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.training_losses.append(loss.cpu().detach().numpy())
        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_loss_epoch",
            np.mean(self.training_losses),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.training_losses = []

    def validation_step(self, batch, batch_idx):

        self.model.eval()
        anchors, positives, negatives = batch

        if not self.from_embeddings:
            if self.mode == "img_retrieval":
                anchors = self.text_encoder(anchors)
                positives = self.image_encoder(positives)
                negatives = self.image_encoder(negatives)
            elif self.mode == "text_retrieval":
                anchors = self.image_encoder(anchors)
                positives = self.text_encoder(positives)
                negatives = self.text_encoder(negatives)

        if self.mode == "img_retrieval":
            anchors = self.text_reprojection(anchors)

        elif self.mode == "text_retrieval":
            positives = self.text_reprojection(positives)
            negatives = self.text_reprojection(negatives)

        loss = self.criterion(anchors, positives, negatives)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_losses.append(loss.cpu().detach().numpy())
        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val_loss_epoch",
            np.mean(self.validation_losses),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.validation_losses = []

    def test_step(self):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = {"scheduler": self.lr_scheduler, "name": "lr_scheduler"}

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        if self.mode == "img_retrieval":
            text_embedding = self.text_encoder(x)
            text_embedding = self.text_reprojection(text_embedding)
            return text_embedding
        elif self.mode == "text_retrieval":
            image_embedding = self.image_encoder(x)
            image_embedding = self.image_reprojection(image_embedding)
            return image_embedding


import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
import torch.nn.functional as F


class ImageTextRetrievalModel(pl.LightningModule):
    def __init__(
        self,
        text_embedding_dim=768,
        image_embedding_dim=2048,
        margin=0.2,
        pretrained_bert="bert-base-uncased",
        mode="text2img",
        learning_rate=1e-3,
    ):
        super().__init__()

        # Load pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the classification layer

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_bert)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert)

        # Projection layers to map image and text embeddings to the same dimension
        self.image_projection = nn.Linear(image_embedding_dim, image_embedding_dim)
        self.text_projection = nn.Linear(text_embedding_dim, image_embedding_dim)

        # Triplet loss margin
        self.margin = margin

        self.mode = mode
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1
        )

    def forward(self, images, texts):
        # Forward pass for images
        image_features = self.resnet(images)
        image_embeddings = self.image_projection(image_features)

        # Forward pass for texts
        input_ids = self.tokenizer(texts, return_tensors="pt", padding=True)[
            "input_ids"
        ]
        text_outputs = self.bert(input_ids=input_ids)
        text_embeddings = self.text_projection(
            text_outputs.last_hidden_state[:, 0, :]
        )  # Use CLS token

        return image_embeddings, text_embeddings

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.cosine_similarity(anchor, positive)
        distance_negative = F.cosine_similarity(anchor, negative)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        images, texts = batch
        anchor_image_embeddings, anchor_text_embeddings = self(images, texts)

        # Randomly select positive and negative samples for triplet loss
        positive_image_embeddings, positive_text_embeddings = self(
            images, texts
        )  # Use same images and texts
        negative_image_embeddings, negative_text_embeddings = self(
            images, texts
        )  # Use same images and texts
        if self.mode == "text2img":
            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_text_embeddings,
                positive_image_embeddings,
                negative_image_embeddings,
            )

        elif self.mode == "img2text":
            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_image_embeddings,
                positive_text_embeddings,
                negative_text_embeddings,
            )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        anchor_image_embeddings, anchor_text_embeddings = self(images, texts)

        images, texts = batch
        anchor_image_embeddings, anchor_text_embeddings = self(images, texts)

        # Randomly select positive and negative samples for triplet loss
        positive_image_embeddings, positive_text_embeddings = self(
            images, texts
        )  # Use same images and texts
        negative_image_embeddings, negative_text_embeddings = self(images, texts)

        if self.mode == "text2img":
            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_text_embeddings,
                positive_image_embeddings,
                negative_image_embeddings,
            )

        elif self.mode == "img2text":
            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_image_embeddings,
                positive_text_embeddings,
                negative_text_embeddings,
            )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",  # adjust the learning rate scheduler's step interval
                "monitor": "val_loss",  # monitor a metric to adjust the learning rate
            },
        }
