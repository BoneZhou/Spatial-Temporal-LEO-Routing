"""Single source of publication styles for the October 23 teacher revision."""
MAIN = (
    ("proposed_gat_lstm_packet", "Proposed", "#D32F2F", "-", "o", "///"),
    ("graphpr_adapted", "GraphPR-adapted", "#1F77B4", "--", "s", "xxx"),
    ("fdr_marl_adapted", "FDR-MARL-adapted", "#2E7D32", "-.", "^", "\\\\\\"),
    ("dqn_ir_fixed_link", "DQN-IR-adapted (fixed-link)", "#7E57C2", ":", "D", "..."),
)
ABLATION = (MAIN[0],
    ("proposed_gat_dqn_packet", "GAT-DQN", *MAIN[1][2:]),
    ("proposed_lstm_dqn_packet", "LSTM-DQN", *MAIN[2][2:]),
    ("proposed_dqn_packet", "DQN", *MAIN[3][2:]))
SCENARIOS = ("S1_non_cross_normal", "S2_cross_normal", "S3_non_cross_ftp", "S4_cross_ftp")
SCENARIO_LABELS = ("S1\n(Local/Normal)", "S2\n(Cross/Normal)",
                   "S3\n(Local/Burst)", "S4\n(Cross/Burst)")
METRICS = (("throughput_mbps", "Throughput (Mbps)", "throughput"),
           ("packet_loss_rate", "Packet loss rate (%)", "loss"),
           ("average_queue_packets", "Average queue length (packets/sat.)", "queue"),
           ("end_to_end_delay_ms", "End-to-end delay (ms)", "delay"))

def configure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({"font.family": "Times New Roman", "font.size": 9,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": .5,
        "savefig.bbox": "tight"})

def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.6)

def line_style(method):
    _, _, color, linestyle, marker, _ = method
    return dict(color=color, linestyle=linestyle, marker=marker, markevery=10,
                linewidth=2, markersize=5.5, markerfacecolor="white", markeredgewidth=1.2)

def save(fig, path, *, eps=True):
    """PDF keeps SD transparency; EPS uses opaque fills because PS has no alpha."""
    fig.savefig(path.with_suffix(".pdf"))
    if eps:
        fig.savefig(path.with_suffix(".eps"))
    fig.savefig(path.with_suffix(".png"), dpi=160)

