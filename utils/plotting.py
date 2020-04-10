import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pypianoroll
import pretty_midi
import torch
from torchvision.utils import make_grid

FONT_SIZE = 12


def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def convert_rgba_to_rgb(rgba):
    row, col, ch = rgba.shape
    if rgba.dtype == 'uint8':
        rgba = rgba.astype('float32') / 255.0
    if ch == 3:
        return rgba
    assert ch == 4
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32')

    rgb[:, :, 0] = r * a + (1.0 - a)
    rgb[:, :, 1] = g * a + (1.0 - a)
    rgb[:, :, 2] = b * a + (1.0 - a)

    return np.asarray(rgb)


def plot_dim(data, target, filename, dim1=0, dim2=1, xlim=None, ylim=None):
    if xlim is not None:
        plt.xlim(-xlim, xlim)
    if ylim is not None:
        plt.ylim(-ylim, ylim)
    plt.scatter(
        x=data[:, dim1],
        y=data[:, dim2],
        c=target,
        s=12,
        linewidths=0,
        cmap="viridis",
        alpha=0.5
    )
    plt.xlabel(f'dimension: {dim1}')
    plt.ylabel(f'dimension: {dim2}')
    plt.colorbar()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()
    img = Image.open(filename)
    img_resized = img.resize((485, 360), Image.ANTIALIAS)
    img = convert_rgba_to_rgb(np.array(img_resized))
    return img


def plot_grad_flow(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.show()


def create_pair_plot(
        data_frame,
        grouping='Model',
        save_path=None,
):
    """
    Method to create pretty looking box plots
    Code courtesy https://cduvallet.github.io/posts/2018/03/boxplots-in-python
    """
    sns.set_style("whitegrid")

    # Define colors for box-plot line
    dark_green = '#2d9c00'
    dark_red = '#f56701'
    light_green = '#a4ec7b'
    light_red = '#f8a173'
    pal = {
        'AR-VAE:Image': light_green,
        'AR-VAE:Music': dark_green,
        r'$\beta$-VAE:Image': light_red,
        r'$\beta$-VAE:Music': dark_red,
    }

    # set hue order
    hue_order = [r'$\beta$-VAE:Image', r'$\beta$-VAE:Music', 'AR-VAE:Image', 'AR-VAE:Music']

    # The boxplot kwargs get passed to matplotlib's boxplot function.
    # Note how we can re-use our lineprops dict to make sure all the lines
    # match. You could also edit each line type (e.g. whiskers, caps, etc)
    # separately.
    boxplot_kwargs = {
        'palette': pal,
        'hue_order': hue_order,
    }

    # And we can plot just like last time
    plt.rcParams.update({'font.size': FONT_SIZE})
    fig, ax = plt.subplots()

    # create pair plot
    g = sns.pairplot(data_frame, hue=grouping, **boxplot_kwargs)
    # Fix legend
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g._legend.remove()
    g.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=4)
    g.fig.subplots_adjust(top=0.95, right=0.98)

    # other cosmetic changes
    sns.despine(left=True)

    # save plot if needed
    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def create_scatter_plot(
        data_frame,
        x_axis='Interpretability',
        y_axis='Reconstruction Accuracy (in %)',
        grouping=r'$\gamma$',
        size=r'$\delta$',
        width=0.5,
        location='lower left',
        save_path=None,
):
    sns.set_style("whitegrid")

    # And we can plot just like last time
    plt.rcParams.update({'font.size': FONT_SIZE})
    fig = plt.figure()
    # ax = plt.gca()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    # create scatter plot
    sns.scatterplot(
        x=x_axis,
        y=y_axis,
        hue=grouping,
        size=size,
        data=data_frame,
        palette="Paired",
        ax=ax,
    )
    plt.plot(0.4, 94.5, 'x', color='k')
    ax.annotate(r'$\beta$-VAE', (0.4, 94.7))
    # sns.despine(left=True, right=True, top=True, bottom=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim((94.0, 98.0))
    plt.xlim((0.35, 0.95))
    # save plot if needed
    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def create_box_plot(
        data_frame,
        x_axis='Metrics',
        y_axis='Score',
        grouping='Model',
        width=0.5,
        location='lower left',
        save_path=None,
):
    """
    Method to create pretty looking box plots
    Code courtesy https://cduvallet.github.io/posts/2018/03/boxplots-in-python
    """
    sns.set_style("whitegrid")

    # Define colors for box-plot line
    dark_green = '#2d9c00'
    dark_red = '#f56701'
    colors = [dark_red, dark_green]
    pal = {'AR-VAE': dark_green, r'$\beta$-VAE': dark_red}

    # Define colors for box-plot fills
    light_green = '#a4ec7b'
    light_red = '#f8a173'
    face_pal = {'AR-VAE': light_green, r'$\beta$-VAE': light_red}

    # set hue order
    hue_order = [r'$\beta$-VAE', 'AR-VAE']

    # Make sure to remove the 'facecolor': 'w' property here, otherwise
    # the palette gets overrided
    boxprops = {'edgecolor': 'k', 'linewidth': 1}
    lineprops = {'color': 'k', 'linewidth': 1}

    # The boxplot kwargs get passed to matplotlib's boxplot function.
    # Note how we can re-use our lineprops dict to make sure all the lines
    # match. You could also edit each line type (e.g. whiskers, caps, etc)
    # separately.
    boxplot_kwargs = {
        'boxprops': boxprops,
        'medianprops': lineprops,
        'whiskerprops': lineprops,
        'capprops': lineprops,
        'width': width,
        'palette': face_pal,
        'hue_order': hue_order
    }
    stripplot_kwargs = {
        'linewidth': 0.6,
        'size': 3,
        'alpha': 0.5,
        'palette': pal,
        'hue_order': hue_order
    }

    # And we can plot just like last time
    plt.rcParams.update({'font.size': FONT_SIZE})
    fig, ax = plt.subplots()

    # create box plot
    sns.boxplot(
        x=x_axis,
        y=y_axis,
        hue=grouping,
        data=data_frame,
        ax=ax,
        # split=True,
        **boxplot_kwargs
    )

    # add line colors
    for i, artist in enumerate(ax.artists):
        if i % 2 == 0:
            col = colors[0]
        else:
            col = colors[1]

        # This sets the color for the main box
        artist.set_edgecolor(col)
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

    # add individual data-points
    sns.stripplot(
        x=x_axis, y=y_axis,
        hue=grouping,
        data=data_frame,
        ax=ax,
        dodge=True,
        jitter=0.1,
        **stripplot_kwargs
    )

    # other cosmetic changes
    sns.despine(left=True)
    ax.legend_.remove()
    ax.xaxis.label.set_visible(False)
    ax.xaxis.grid(True)

    # Fix the legend, keep only the first two legend elements
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles[0:2],
        labels[0:2],
        loc=location,
        fontsize='large',
        handletextpad=0.5
    )
    lgd.legendHandles[0]._sizes = [40]
    lgd.legendHandles[1]._sizes = [40]

    # save plot if needed
    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def plot_pianoroll_from_midi(midi_path, attr_labels, attr_str, type):
    pr_a = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pr_a.get_piano_roll().astype('int').T
    beat_resolution = 100
    if len(pr_a.instruments) == 0:
        return
    note_list = pr_a.instruments[0].notes
    num_measures = int(piano_roll.shape[0] / (2 * beat_resolution))
    downbeats = [i * 2 * beat_resolution for i in range(num_measures)]

    shaded_piano_roll = np.zeros_like(piano_roll)
    for i in range(0, shaded_piano_roll.shape[1], 2):
        shaded_piano_roll[:, i] = 30
    for i in range(0, shaded_piano_roll.shape[0], 25):
        shaded_piano_roll[i, :] = 50
    for note in note_list:
        start = int(note.start * beat_resolution)
        pitch = int(note.pitch)
        piano_roll[start:start+5, pitch-1:pitch+2] = 127
    shaded_piano_roll[piano_roll != 0] = piano_roll[piano_roll != 0]

    figsize = (16, 2)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [6, 1]})
    pypianoroll.plot_pianoroll(
        ax1,
        shaded_piano_roll,
        downbeats=downbeats,
        beat_resolution=2 * beat_resolution,
        xtick='beat',
    )
    f.set_facecolor('white')
    if type == 'folk':
        ax1.set_ylim(55, 84)
    elif type == 'bach':
        ax1.set_ylim(55, 90)
    ax1.set_ylabel('Pitch')
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    plt.tight_layout()
    ax1.set_xlabel('')
    save_path = os.path.join(
        os.path.dirname(midi_path),
        f'{os.path.splitext(os.path.basename(midi_path))[0]}.png'
    )
    x = [n + 1 for n in range(attr_labels.size)]
    # ax2.bar(x, attr_labels, color='k')
    ax2.plot(x, attr_labels, 'o', color='k', markersize=7)
    ax2.set_ylabel(attr_str)
    # if attr_str == 'contour':
    #     ax2.set_ylim(-0.7, 0.7)
    # else:
    #     ax2.set_ylim(-0.1, 0.5)
    ax2.set_yticklabels([])
    ax2.set_xticks(np.arange(1, num_measures+1))
    plt.savefig(save_path, dpi=500)
    plt.close()


def save_gif(image_tensor, save_filepath, delay=100):
    """
        Convert a tensor containing multiple images to a GIF
    """
    image_list = []
    tensor = torch.cat((image_tensor, image_tensor, image_tensor), 1)
    for n in range(image_tensor.shape[0]):
        ndarr = tensor[n].mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        image_list.append(Image.fromarray(ndarr))
    save_gif_from_list(image_list, save_filepath, delay)


def save_gif_from_list(image_list, save_filepath, delay=100):
    """
        Convert a tensor containing multiple images to a GIF
    """
    image_list[0].save(
        save_filepath, save_all=True, append_images=image_list[1:], optimize=False, duration=delay, loop=0
    )
