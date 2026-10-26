"""Build report.pdf (2 two-column pages + references) from results/summary.csv."""
import csv
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                                Image, Table, TableStyle, FrameBreak, PageBreak, NextPageTemplate)

GITHUB_URL = ""   # <- paste your repository link here and re-run: python make_report.py

ROOT = Path(__file__).resolve().parent
rows = {r["policy"]: r for r in csv.DictReader(open(ROOT / "results" / "summary.csv"))}
f = lambda p, k, d=2: f"{float(rows[p][k]):.{d}f}"

body = ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, leading=10.8, spaceAfter=4)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=5, spaceAfter=3,
                   textColor=colors.HexColor("#1f3a5f"))
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=1)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9, leading=11, alignment=1, textColor=colors.grey)
cap = ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=7.5, leading=9, textColor=colors.HexColor("#444444"), spaceAfter=5)
meta = ParagraphStyle("m", parent=body, fontSize=8.3, leading=10.2)

W, H = A4
M = 0.6 * inch
colw = (W - 2 * M - 0.25 * inch) / 2
top = 1.05 * inch
frames_p1 = [Frame(M, M + 0.2*inch, colw, H - M - top - 0.2*inch, id="l1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
             Frame(M + colw + 0.25*inch, M + 0.2*inch, colw, H - M - top - 0.2*inch, id="r1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
frames_p2 = [Frame(M, M + 0.2*inch, colw, H - 2*M - 0.2*inch, id="l2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
             Frame(M + colw + 0.25*inch, M + 0.2*inch, colw, H - 2*M - 0.2*inch, id="r2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
frame_ref = [Frame(M, M, W - 2*M, H - 2*M, id="ref")]

def footer(c, doc):
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(M, 0.4*inch, "AI Assignment 2 | RL-Based Smart Parking Allocation")
    c.drawRightString(W - M, 0.4*inch, f"Page {doc.page}")

def first_page(c, doc):
    footer(c, doc)

doc = BaseDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, title="RL-Based Smart Parking Allocation",
                      author="Duggirala Manoj, Likith Reddy")
doc.addPageTemplates([PageTemplate("p1", frames_p1, onPage=first_page),
                      PageTemplate("p2", frames_p2, onPage=footer),
                      PageTemplate("ref", frame_ref, onPage=footer)])

def P(t, s=body): return Paragraph(t, s)
def img(name, w=colw): 
    from PIL import Image as PI
    iw, ih = PI.open(ROOT / "figures" / name).size
    return Image(str(ROOT / "figures" / name), width=w, height=w * ih / iw)

story = []
# header drawn into first frame region using a spanning table is awkward; put it as flowables before first column via canvas
def header(c, doc_):
    pass

# ---- Header (page 1 top, full width) via a one-time canvas callback ----
def page1(c, doc_):
    footer(c, doc_)
    c.setFont("Helvetica-Bold", 16); c.setFillColor(colors.black)
    c.drawCentredString(W/2, H - 0.75*inch, "RL-Based Smart Parking Allocation")
    c.setFont("Helvetica", 9); c.setFillColor(colors.grey)
    c.drawCentredString(W/2, H - 0.93*inch, "AI Assignment 2 \u2014 Real-Life Use Cases of Reinforcement Learning")
doc.pageTemplates[0].onPage = page1

gh = GITHUB_URL if GITHUB_URL else "(add repository link)"
story += [
 P("<b>Team:</b> Duggirala Manoj (B24CM1022, b24cm1022@iitj.ac.in) and Likith Reddy (B24CM1069, b24cm1069@iitj.ac.in)<br/><b>GitHub:</b> " + gh, meta),
 P("Abstract", h),
 P(open(ROOT / "project_abstract_50_words.txt").read().strip()),
 P("1. Problem and objective", h),
 P("Parking allocation is difficult when vehicles arrive continuously, spaces have different walking distances, and parked vehicles leave at uncertain times. A fixed rule can overload one zone while others stay free. We model a small three-zone car park and train a tabular Q-learning agent to assign each arriving vehicle to a zone, aiming to reduce simulated search effort and congestion while keeping walking distance reasonable."),
 P("2. Environment", h),
 P("The custom Python simulator has three zones with capacities 5, 6 and 7 and walking-distance costs 1.8, 1.0 and 3.2 units. One vehicle arrives per time step with a short, medium or long stay (2\u20135, 6\u201312 or 13\u201324 steps). The stay mix changes across the day, and parked vehicles' remaining times decrease each step. An episode has 120 vehicle decisions."),
 P("3. RL formulation", h),
 P("<b>State:</b> three quantized zone-occupancy levels (0\u20133 each), vehicle stay category (0\u20132) and time segment (0\u20132): at most 4<super>3</super>\u00d73\u00d73 = 576 states. <b>Action:</b> choose one of the three zones; full zones are masked, and the environment penalizes an invalid choice. <b>Transition:</b> the vehicle's stay is added to the chosen zone, time advances, and all remaining stays drop by one."),
 P("4. Reward function", h),
 P("For a valid assignment: <b>r = 5 \u2212 0.8S \u2212 0.45W \u2212 0.9C \u2212 D</b>, where S is simulated search time (rising with pre-assignment occupancy), W the zone walking distance, C a congestion penalty above 65% occupancy and D a small expected-stay blocking cost. An invalid action receives \u221212. The weights are simulator design choices, not measured human preferences."),
 P("5. Learning algorithm", h),
 P("Tabular Q-learning with epsilon-greedy exploration: Q(s,a) \u2190 Q(s,a) + \u03b1[r + \u03b3 max Q(s\u2032,a\u2032) \u2212 Q(s,a)], with no future term at terminal steps. We use \u03b1 = 0.15, \u03b3 = 0.93, epsilon decaying by 0.992 per episode to a minimum of 0.04, 1,800 training episodes and seed 7."),
 FrameBreak(),
 P("6. Baselines and evaluation", h),
 P("The trained policy is evaluated over 150 seeded episodes (seeds from 2000) against three baselines: first-available, nearest-available and least-occupied. All policies see the same episode seeds. Metrics: average search time, walking distance and congestion penalty per accepted vehicle, acceptance rate and invalid-action rate."),
 P("7. Results", h),
]
order = ["Q-learning", "First available", "Nearest available", "Least occupied"]
tdata = [["Policy", "Search", "Walk", "Congest.", "Accept"]] + [
    [p, f(p, "avg_search_time"), f(p, "avg_walking_distance"), f(p, "avg_congestion_penalty", 3), f(p, "acceptance_rate")] for p in order]
t = Table(tdata, colWidths=[colw*0.34, colw*0.16, colw*0.14, colw*0.19, colw*0.17])
t.setStyle(TableStyle([("FONT", (0,0), (-1,0), "Helvetica-Bold", 7.6), ("FONT", (0,1), (-1,-1), "Helvetica", 7.6),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6ecf5")), ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
    ("ALIGN", (1,0), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]))
story += [
 P(f"Q-learning reached an average search cost of {f('Q-learning','avg_search_time')}, versus {f('First available','avg_search_time')} (first-available), {f('Nearest available','avg_search_time')} (nearest-available) and {f('Least occupied','avg_search_time')} (least-occupied). Mean walking distance was {f('Q-learning','avg_walking_distance')}, {f('First available','avg_walking_distance')}, {f('Nearest available','avg_walking_distance')} and {f('Least occupied','avg_walking_distance')}. Every policy accepted all arrivals, so acceptance rate does not separate them."),
 t, P("Table 1. Mean metrics over 150 evaluation episodes (search and walk in simulator units).", cap),
 img("training_curve.png"), P("Figure 1. Q-learning reward during training (60-episode moving average).", cap),
 NextPageTemplate("p2"), PageBreak(),
 P("8. Policy comparison", h),
 P("The least-occupied baseline has slightly lower search cost than Q-learning in this run, while Q-learning cuts search cost by roughly 42% against first-available and 44% against nearest-available. The nearest- and first-available rules have the shortest walks but the highest congestion, whereas Q-learning and least-occupied spread load more evenly at the price of longer walks. The RL agent therefore does not beat every baseline on every metric; this is a small simulation study."),
 img("avg_search_time.png"), P("Figure 2. Average simulated search cost across policies.", cap),
 img("avg_walking_distance.png"), P("Figure 3. Average walking distance across policies.", cap),
 FrameBreak(),
 P("9. Limitations", h),
 P("The environment is synthetic: stylized arrivals, stays, capacities and hand-designed reward weights. It uses no real parking data, models no individual routes and has no real-time sensors. No policy rejected a vehicle, so demand pressure should be raised in future tests. Results come from a single training seed."),
 P("10. Conclusion and next steps", h),
 P("We built a reproducible RL formulation for parking-zone assignment with a custom simulator, tabular Q-learning, three baselines, saved Q-table, CSV records and plots. The results show a trade-off between search effort and walking distance but do not establish superiority over rule-based methods. Future work: sensitivity tests over capacities and arrival rates, multiple training seeds, and calibration with permitted real parking data."),
 P("Reproduction", h),
 P("Install Python 3.10+ and <b>requirements.txt</b>, then run <b>python run_experiments.py</b> to train, evaluate, save the Q-table, write CSVs and regenerate figures (training seed 7, evaluation seeds from 2000). Contents: parking_env.py, rl_agent.py, run_experiments.py, trained_q_table.pkl, results/*.csv, figures/*.png, README.md and this report."),
 NextPageTemplate("ref"), PageBreak(),
 P("References", h),
 P("1. Sutton, R. S., &amp; Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i> (2nd ed.). MIT Press."),
 P("2. Watkins, C. J. C. H., &amp; Dayan, P. (1992). Q-learning. <i>Machine Learning</i>, 8, 279\u2013292. https://doi.org/10.1007/BF00992698"),
 P("3. The simulator, reward function and experiment code were implemented for this project; all reported values are generated by run_experiments.py with the documented seeds."),
 P("4. No external RL repository is used as the implementation base; the cited sources are conceptual references for the method."),
]
doc.build(story)
