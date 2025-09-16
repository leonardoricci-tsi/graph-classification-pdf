import os, io, csv, argparse, re
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO
import pytesseract
from datetime import datetime

MODEL_PATH = "weights/best_charts.pt"  
OUT_DIR = Path("agro_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_KWS = {
    "NDVI/Índice de vegetação": ["ndvi","índice de vegetação","vegetation index","evi","sav"],
    "Clima/Chuva/Temperatura": ["chuva","precipitação","mm","pluviometria","temperatura","anomalia","el niño","la niña"],
    "Mercado/Preços/Commodities": ["preço","cotação","usd","r$/saca","future","futuros","cbot","soja","milho","caf\u00e9","trigo"],
    "Produção/Rendimento": ["rendimento","produtividade","t/ha","sacas/ha","produção","yield","harvest"],
    "Sanidade/Doenças/Pragas": ["ferrugem","míldio","mancha","lagarta","percevejo","incidência","severidade"],
}

TREND_KWS = {
    "alta": ["aumento","alta","subida","crescimento","uptrend","rally"],
    "queda": ["queda","redução","baixa","recuo","downtrend","declínio"],
    "volátil": ["volátil","oscilação","flutuação","volatility"]
}

def page_image(page, dpi=220):
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img, lang="por+eng").strip()
    except Exception:
        return ""

def summarize(text: str, limit=220):
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    return clean[:limit] + ("..." if len(clean) > limit else "")

def guess_topics(text: str, chart_label: str):
    t = (text or "").lower()
    hits = [topic for topic,kws in TOPIC_KWS.items() if any(k in t for k in kws)]
    if not hits:
        if chart_label in ("heatmap","map","choropleth"):
            hits.append("Clima/Chuva/Temperatura")
        elif chart_label in ("candlestick","line_graph"):
            hits.append("Mercado/Preços/Commodities")
    return list(set(hits)) or ["Outro"]

def guess_trend(text: str):
    t = (text or "").lower()
    for k in TREND_KWS["alta"]:
        if k in t: return "tendência de alta (texto)"
    for k in TREND_KWS["queda"]:
        if k in t: return "tendência de queda (texto)"
    for k in TREND_KWS["volátil"]:
        if k in t: return "volátil/oscilando (texto)"
    return "indefinida"

def process_pdf(pdf_path: str, model: YOLO, conf_th=0.35):
    rows = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # renderiza página
            pimg = page_image(page)
            page_text = (page.get_text() or "") + "\n" + ocr_text(pimg)

            # inferência
            results = model.predict(pimg, conf=conf_th)

            for r in results:
                # --- CASO 1: MODELO DE DETECÇÃO (tem caixas) ---
                if getattr(r, "boxes", None) is not None and len(r.boxes):
                    for b in r.boxes:
                        cls = model.names[int(b.cls)]
                        conf = float(b.conf)
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        crop = pimg.crop((x1, y1, x2, y2))
                        crop_name = f"p{i+1:03d}_{cls}_{x1}_{y1}.png"
                        crop_path = OUT_DIR / crop_name
                        crop.save(crop_path)

                        local_txt = ocr_text(crop)
                        topics = "; ".join(guess_topics(page_text + " " + local_txt, cls))
                        trend = guess_trend(page_text + " " + local_txt) if cls in ("line_graph","candlestick","line_chart") else "n/a"
                        rows.append({
                            "arquivo_pdf": os.path.basename(pdf_path),
                            "pagina": i+1,
                            "tipo_grafico": cls,
                            "confianca": round(conf, 3),
                            "topicos": topics,
                            "tendencia_hint": trend,
                            "texto_resumo": summarize(local_txt or page_text),
                            "recorte": crop_name,
                            "topk": ""  # só pra manter colunas consistentes
                        })

                # --- CASO 2: MODELO DE CLASSIFICAÇÃO (sem caixas) ---
                elif getattr(r, "probs", None) is not None:
                    # scores por classe
                    scores = r.probs.data.tolist()  
                    names = model.names
                    topk = sorted([(names[i], float(scores[i])) for i in range(len(scores))],
                                  key=lambda x: x[1], reverse=True)[:3]
                    top1_label, top1_conf = topk[0]

                    # salva a página inteira como "recorte"
                    crop_name = f"p{i+1:03d}_{top1_label}.png"
                    (OUT_DIR / crop_name).parent.mkdir(parents=True, exist_ok=True)
                    pimg.save(OUT_DIR / crop_name)

                    topics = "; ".join(guess_topics(page_text, top1_label))
                    trend = guess_trend(page_text) if top1_label in ("line_graph","line_chart","candlestick") else "n/a"

                    rows.append({
                        "arquivo_pdf": os.path.basename(pdf_path),
                        "pagina": i+1,
                        "tipo_grafico": top1_label,
                        "confianca": round(top1_conf, 3),
                        "topicos": topics,
                        "tendencia_hint": trend,
                        "texto_resumo": summarize(page_text),
                        "recorte": crop_name,
                        "topk": "; ".join([f"{lbl}:{conf:.2f}" for lbl, conf in topk])
                    })

                # --- CASO 3: nada detectado/classificado acima do threshold ---
                else:
                    # opcional: ignore a página silenciosamente
                    continue
    return rows


def write_csv(rows, path):
    if not rows: return
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def write_md(rows, path):
    from collections import defaultdict
    by_topic = defaultdict(list)
    for r in rows:
        for t in r["topicos"].split("; "):
            by_topic[t].append(r)
    lines = []
    lines.append(f"# Relatório Agro — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    fontes = ", ".join(sorted(set(r["arquivo_pdf"] for r in rows)))
    lines.append(f"**Fontes:** {fontes}\n")
    for topic, items in by_topic.items():
        lines.append(f"## {topic} ({len(items)})")
        for it in items[:12]:
            lines.append(f"- p.{it['pagina']} • **{it['tipo_grafico']}** (conf. {it['confianca']}) • tendência: {it['tendencia_hint']} • `{it['recorte']}`")
            lines.append(f"  - _Hint:_ {it['texto_resumo']}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="+", required=True, help="Caminhos dos PDFs (Embrapa/Conab/USDA etc.)")
    ap.add_argument("--weights", default=MODEL_PATH, help="Peso YOLO de gráficos")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    model = YOLO(args.weights)
    all_rows = []
    for pdf in args.pdfs:
        print(f"[+] {pdf}")
        all_rows += process_pdf(pdf, model, conf_th=args.conf)

    csv_path = OUT_DIR / "insights.csv"
    md_path  = OUT_DIR / "relatorio.md"
    write_csv(all_rows, csv_path)
    write_md(all_rows, md_path)
    print(f"\nOK → {csv_path}\nOK → {md_path}\nRecortes em {OUT_DIR}/")

if __name__ == "__main__":
    main()
PY
