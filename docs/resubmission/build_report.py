# -*- coding: utf-8 -*-
"""Build a self-contained HTML report from methodology-draft.md + figures."""
import base64, re, markdown
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
md_text = (HERE / 'methodology-draft.md').read_text()

# drop the internal H1 + editor blockquote; keep from Abstract on
body_md = md_text[md_text.index('## Abstract'):]
html = markdown.markdown(body_md, extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list'])

# clean anchor ids on the section headers we navigate to
for text, an in [('Abstract', 'abstract'), ('Contributions', 'contributions'),
                 ('Part A', 'methods'), ('Part B', 'results'), ('Part C', 'discussion')]:
    html = re.sub(r'<h2[^>]*>(' + re.escape(text) + r')', r'<h2 id="' + an + r'">\1', html, count=1)

# highlight baseline / proposed-model rows in tables
html = re.sub(r'<tr>(\s*<td[^>]*>(?:<[^>]+>|\s|\*)*Demographics)', r'<tr class="row-demo">\1', html)
html = re.sub(r'<tr>(\s*<td[^>]*>(?:<[^>]+>|\s|\*)*NeuroFormer)', r'<tr class="row-nf">\1', html)

# wrap tables for horizontal scroll
html = html.replace('<table>', '<div class="tw"><table>').replace('</table>', '</table></div>')

# figures as data URIs, inserted before the next section heading
def fig(name, cap):
    b = base64.b64encode((ROOT / 'figures' / name).read_bytes()).decode()
    return (f'<figure><img alt="{cap}" src="data:image/png;base64,{b}">'
            f'<figcaption>{cap}</figcaption></figure>')

INSERTS = {
    r'(<h3[^>]*>A\.7 )': fig('fig2_architecture.png',
        'Figure 2. NeuroFormer architecture — modality-specific encoders and confidence-gated '
        'fusion with a masked missing-modality path. Behaviour is greyed: unused in this cohort.'),
    r'(<h3[^>]*>B\.2 )': fig('fig1_benchmark_forest.png',
        'Figure 1. Leakage-free benchmark (MIPDB, N=92). Every model’s bootstrap CI overlaps '
        'the age+sex baseline (gold band); NeuroFormer in red sits mid-pack.'),
    r'(<h3[^>]*>B\.3 )': (fig('fig3_leakage.png',
        'Figure 3. Window-level (leaky) evaluation inflates AUC toward 1.0; honest subject-level '
        'evaluation returns ~0.5.') + fig('fig4_confound_collapse.png',
        'Figure 4. Removing the age+sex difference by joint matching collapses the signal '
        '(demographics 0.71 → 0.40).')),
    r'(<h3[^>]*>B\.5 )': fig('fig5_cross_cohort.png',
        'Figure 5. EEG sits below demographics in the confounded paediatric cohort but above it '
        'in the independent adult cohort.'),
}
for pat, f in INSERTS.items():
    html = re.sub(pat, f + r'\1', html, count=1)

STYLE = """
<style>
:root{
  --ground:#F6F8FA; --panel:#FFFFFF; --ink:#1A2530; --muted:#5A6B78;
  --accent:#14646E; --accent-soft:#E3EEEF; --gold:#9C7211; --gold-soft:#FBF3DD;
  --red:#B23A2E; --line:#E1E7EC;
  --serif:"Iowan Old Style",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--serif);font-size:18px;line-height:1.66;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
.wrap{display:grid;grid-template-columns:15rem minmax(0,44rem);gap:3.5rem;
  max-width:70rem;margin:0 auto;padding:3rem 2rem 6rem;}
@media(max-width:820px){.wrap{grid-template-columns:1fr;gap:1.5rem;padding:1.5rem}}

/* nav */
nav{position:sticky;top:2rem;align-self:start;font-family:var(--sans);font-size:.8rem;}
@media(max-width:820px){nav{position:static;border-bottom:1px solid var(--line);padding-bottom:1rem}}
nav .brand{font-weight:700;letter-spacing:.02em;color:var(--ink);font-size:.92rem;margin-bottom:1rem;line-height:1.3}
nav a{display:block;color:var(--muted);text-decoration:none;padding:.32rem 0;border-left:2px solid transparent;padding-left:.9rem;margin-left:-.9rem;transition:color .15s,border-color .15s}
nav a:hover{color:var(--accent);border-color:var(--accent)}
nav a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
nav .grp{text-transform:uppercase;letter-spacing:.12em;font-size:.66rem;color:var(--muted);margin:1.4rem 0 .3rem}

main{min-width:0}
.eyebrow{font-family:var(--sans);text-transform:uppercase;letter-spacing:.14em;
  font-size:.72rem;font-weight:600;color:var(--accent);}
h1.title{font-size:2.15rem;line-height:1.15;margin:.4rem 0 .5rem;text-wrap:balance;font-weight:700;letter-spacing:-.01em}
.lede{color:var(--muted);font-size:1.05rem;margin:0 0 1.6rem}
.meta{display:flex;flex-wrap:wrap;gap:.5rem;font-family:var(--sans);margin-bottom:2rem}
.chip{font-size:.74rem;letter-spacing:.02em;background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:.28rem .7rem;color:var(--ink)}
.chip b{color:var(--accent)}

/* headline stats */
.headline{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:8px;padding:1.3rem 1.5rem;margin:0 0 2.4rem;
  display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;font-family:var(--sans)}
@media(max-width:560px){.headline{grid-template-columns:1fr;gap:.9rem}}
.headline .stat{border-left:1px solid var(--line);padding-left:1rem}
.headline .stat:first-child{border-left:0;padding-left:0}
.headline .num{font-family:var(--serif);font-size:1.7rem;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.headline .num.gold{color:var(--gold)} .headline .num.red{color:var(--red)} .headline .num.teal{color:var(--accent)}
.headline .cap{font-size:.76rem;color:var(--muted);margin-top:.4rem;line-height:1.4}

h2{font-size:1.55rem;font-weight:700;margin:3rem 0 .3rem;padding-top:1.6rem;
  border-top:1px solid var(--line);letter-spacing:-.01em;text-wrap:balance}
h2:first-of-type{border-top:0;padding-top:0;margin-top:1rem}
h3{font-family:var(--sans);font-size:1.06rem;font-weight:700;margin:2.1rem 0 .4rem;color:var(--ink);letter-spacing:-.005em;line-height:1.3}
p{margin:.85rem 0}
strong{font-weight:700}
em{font-style:italic}
a{color:var(--accent)}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.86em;
  background:var(--accent-soft);color:#0f4a52;padding:.08em .38em;border-radius:4px}
ul,ol{padding-left:1.2rem;margin:.85rem 0}
li{margin:.4rem 0}
hr{border:0;border-top:1px solid var(--line);margin:2.2rem 0}
blockquote{border-left:3px solid var(--accent);background:var(--accent-soft);
  margin:1.2rem 0;padding:.6rem 1.1rem;color:#123;border-radius:0 6px 6px 0;font-size:.95rem}

/* tables */
.tw{overflow-x:auto;margin:1.3rem 0;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.85rem;
  font-variant-numeric:tabular-nums}
thead th{background:var(--accent-soft);color:#0f4a52;text-align:left;font-weight:600;
  letter-spacing:.02em;padding:.6rem .8rem;border-bottom:1px solid var(--line);white-space:nowrap}
tbody td{padding:.5rem .8rem;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:0}
tbody td:not(:first-child){text-align:right}
tr.row-demo td{background:var(--gold-soft);font-weight:700}
tr.row-demo td:first-child{box-shadow:inset 3px 0 0 var(--gold)}
tr.row-nf td{background:#F4F6F8;font-weight:700}

/* figures */
figure{margin:1.8rem 0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:1rem;text-align:center}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-family:var(--sans);font-size:.78rem;color:var(--muted);
  line-height:1.5;text-align:left;margin-top:.8rem;padding-top:.7rem;border-top:1px solid var(--line)}

footer{font-family:var(--sans);font-size:.76rem;color:var(--muted);margin-top:3rem;
  padding-top:1.2rem;border-top:1px solid var(--line)}
</style>
"""

HEADER = """
<header>
  <div class="eyebrow">Methodology &amp; results · review draft</div>
  <h1 class="title">A leakage-free multimodal evaluation of NeuroFormer</h1>
  <p class="lede">EEG + eye-tracking psychiatric screening, evaluated honestly across two cohorts.</p>
  <div class="meta">
    <span class="chip">Cohort <b>MIPDB N=92</b> (20 positive)</span>
    <span class="chip"><b>14</b> models, one protocol</span>
    <span class="chip">External <b>ds003478</b> (N≈121)</span>
    <span class="chip">Framing <b>rigor + architecture</b></span>
  </div>
  <div class="headline">
    <div class="stat"><div class="num gold">0.737</div><div class="cap">Best model is the age+sex baseline — no model beats it</div></div>
    <div class="stat"><div class="num teal">0 / 91</div><div class="cap">Pairwise model differences significant after FDR correction</div></div>
    <div class="stat"><div class="num red">0.51→0.98</div><div class="cap">AUC under honest vs window-level (leaky) evaluation</div></div>
  </div>
</header>
"""

NAV = """
<nav aria-label="Contents">
  <div class="brand">NeuroFormer<br>evaluation report</div>
  <a href="#abstract">Abstract</a>
  <a href="#contributions">Contributions</a>
  <div class="grp">Part A — Methods</div>
  <a href="#methods">Cohorts, features, model, protocol</a>
  <div class="grp">Part B — Results</div>
  <a href="#results">Benchmark, confounds, cross-cohort</a>
  <div class="grp">Part C</div>
  <a href="#discussion">Discussion</a>
</nav>
"""

FOOTER = ('<footer>Generated from <code>methodology-draft.md</code> and <code>results/*.csv</code>. '
          'All numbers are the standardized reference-suite runs; NeuroFormer is reported as evaluated '
          'here (128-d handcrafted EEG + eye), not the original 516-d/3-modality pipeline.</footer>')

TITLE = '<title>NeuroFormer — leakage-free evaluation</title>'
out = TITLE + STYLE + '<div class="wrap">' + NAV + '<main>' + HEADER + html + FOOTER + '</main></div>'
(HERE / 'report.html').write_text(out)
print('wrote', HERE / 'report.html', '(%.0f KB)' % (len(out)/1024))
