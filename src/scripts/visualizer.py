import io
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import logging
import flwr as fl

# Implement this all with a singleton class?
# It already kind of is (a module with functions and variables) --- see 
# https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

summary_writer = None
def init_summary_writer(path):
    global summary_writer
    summary_writer = SummaryWriter(log_dir=path, filename_suffix='tb')

def visualize_model(model, data):
    summary_writer.add_graph(model, data)

def summarize_weight_update(tag, flat_weights, step):
    summary_writer.add_scalar(tag+'/mean', flat_weights.mean(), step)
    summary_writer.add_scalar(tag+'/variance', flat_weights.var(), step)
    summary_writer.add_scalar(tag+'/min', flat_weights.min(), step)
    summary_writer.add_scalar(tag+'/max', flat_weights.max(), step)
    # Only compute histogram every 20 rounds for performance reasons.
    if step % 20 == 0:
        fig, ax = plt.subplots()
        if summarize_weight_update.BIN_EDGES is None:
            summarize_weight_update.BIN_EDGES = np.histogram_bin_edges(flat_weights, 'sturges')
        ax.hist(flat_weights, bins=summarize_weight_update.BIN_EDGES)
        percents = [5, 10, 90, 95]
        quantiles = np.percentile(flat_weights, percents)
        for q,p in zip(quantiles, percents):
            ax.axvline(q, label=f"{p}th percentile", linestyle=":", color="red")
        buf = io.BytesIO()
        fig.savefig(buf, format='jpeg')
        plt.close(fig)
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        summary_writer.add_image(tag+'/histogram', image, step)
summarize_weight_update.BIN_EDGES = None

def summarize_loss(tag, loss, step):
    summary_writer.add_scalar(tag+'/loss', loss, step)
    fl.common.logger.log(logging.INFO, f"Loss: {loss}")

def summarize_other(tag, loss, step):
    summary_writer.add_scalar(tag, loss, step)

def summarize_accuracy(tag, accuracy, step):
    summary_writer.add_scalar(tag+'/accuracy', accuracy, step)
    fl.common.logger.log(logging.INFO, f"Accuracy: {accuracy}")

sent_bytes = []
def add_bytes(bytes):
    global sent_bytes
    sent_bytes.append(bytes)

sent_bytes_downlink = []
def add_bytes_downlink(bytes):
    global sent_bytes_downlink
    sent_bytes_downlink.append(bytes)

# test_accuracies = []
# test_rounds = []
# def add_test_accuracy(accuracy, rnd):
#     global test_accuracies
#     global test_rounds
#     test_accuracies.append(accuracy)
#     test_rounds.append(rnd)

def summarize_bytes_sent_per_round(tag, bytes, step):
    summary_writer.add_scalar(tag, bytes/1_000_000, step)

def summarize_bytes_sent_total(tag, step):
    summary_writer.add_scalar(tag, sum(sent_bytes)/1_000_000, step)

def summarize_bytes_sent_per_round_downlink(tag, bytes, step):
    summary_writer.add_scalar(tag, bytes/1_000_000, step)

def summarize_bytes_sent_total_downlink(tag, step):
    summary_writer.add_scalar(tag, sum(sent_bytes_downlink)/1_000_000, step)


# def summarize_accuracy_over_bytes_sent(tag, step):
#     fig, ax = plt.subplots()
#     # Ignore accuracies[0] because it is evaluated before any communication
#     # has happened.
#     ax.plot(np.cumsum(sent_bytes)/1_000_000, test_accuracies)
#     summary_writer.add_figure(tag, fig, step)
