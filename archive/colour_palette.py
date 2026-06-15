# Define TLG Brand Colours
tlg_colours = {
    'background': '#faf7f8',
    'text': '#313130',
    'lines': ['#1a2792', '#ffb7ff', '#c7ef00', '#f7574b', '#21fa90', '#7f96ff']
}


def apply_theme():
    """Apply the TLG brand colour palette to matplotlib rcParams."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.facecolor': tlg_colours['background'],
        'axes.facecolor': tlg_colours['background'],
        'axes.edgecolor': tlg_colours['text'],
        'axes.labelcolor': tlg_colours['text'],
        'xtick.color': tlg_colours['text'],
        'ytick.color': tlg_colours['text'],
        'text.color': tlg_colours['text'],
        'axes.prop_cycle': plt.cycler(color=tlg_colours['lines']),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })

