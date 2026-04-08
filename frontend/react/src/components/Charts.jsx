import { useEffect, useRef } from "react";

function drawAxes(ctx, w, h, pad) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
}

export function LineChart({ seriesA, seriesB, labelA = "A", labelB = "B" }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 30;
    drawAxes(ctx, w, h, pad);

    const all = [...seriesA, ...seriesB];
    if (!all.length) return;
    const yMax = Math.max(...all, 1);

    const draw = (arr, color) => {
      if (!arr.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      const stepX = (w - pad * 2) / Math.max(arr.length - 1, 1);
      ctx.beginPath();
      arr.forEach((v, i) => {
        const x = pad + i * stepX;
        const y = h - pad - (Math.max(0, v) / yMax) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    draw(seriesA, "#fff");
    draw(seriesB, "#8c8c8c");
    ctx.fillStyle = "#bcbcbc";
    ctx.font = "12px Segoe UI";
    ctx.fillText(labelA, 40, 20);
    ctx.fillText(labelB, 120, 20);
  }, [seriesA, seriesB, labelA, labelB]);

  return <canvas className="chart-canvas" ref={ref} width={1000} height={280} />;
}

export function CompareBars({ rows }) {
  return (
    <div className="compare-bars">
      {rows.map((row) => (
        <div key={row.label} className="compare-row">
          <div className="compare-label">{row.label}</div>
          <div className="compare-track">
            <div className="compare-fill" style={{ width: `${Math.max(0, Math.min(100, row.value * 100))}%` }} />
          </div>
          <div className="compare-value">{row.value.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

