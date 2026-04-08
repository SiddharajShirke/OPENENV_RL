import { useEffect, useRef } from "react";

function drawGridAndAxes(ctx, w, h, pad, yMin, yMax) {
  const chartW = w - pad * 2;
  const chartH = h - pad * 2;
  ctx.clearRect(0, 0, w, h);

  // chart area background
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#060b12");
  bg.addColorStop(1, "#03070d");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#13202f";
  ctx.lineWidth = 1;
  const gridRows = 5;
  for (let i = 0; i <= gridRows; i += 1) {
    const y = pad + (chartH * i) / gridRows;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  const gridCols = 8;
  for (let i = 0; i <= gridCols; i += 1) {
    const x = pad + (chartW * i) / gridCols;
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, h - pad);
    ctx.stroke();
  }

  ctx.strokeStyle = "#2a3e54";
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  const zeroInRange = yMin <= 0 && yMax >= 0;
  if (zeroInRange) {
    const yRange = Math.max(1e-9, yMax - yMin);
    const y0 = pad + ((yMax - 0) / yRange) * chartH;
    ctx.strokeStyle = "#2d5f84";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, y0);
    ctx.lineTo(w - pad, y0);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

export function LineChart({ seriesA, seriesB, labelA = "A", labelB = "B" }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 40;

    const all = [...seriesA, ...seriesB];
    if (!all.length) return;
    const yMaxRaw = Math.max(...all);
    const yMinRaw = Math.min(...all);
    const margin = Math.max(1, (yMaxRaw - yMinRaw) * 0.12);
    const yMax = yMaxRaw + margin;
    const yMin = yMinRaw - margin;
    const yRange = Math.max(1e-9, yMax - yMin);
    const chartW = w - pad * 2;
    const chartH = h - pad * 2;

    drawGridAndAxes(ctx, w, h, pad, yMin, yMax);

    const yPx = (value) => pad + ((yMax - value) / yRange) * chartH;

    const draw = (arr, color, glowColor) => {
      if (!arr.length) return;
      ctx.shadowBlur = 8;
      ctx.shadowColor = glowColor;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.25;
      const stepX = chartW / Math.max(arr.length - 1, 1);
      ctx.beginPath();
      arr.forEach((v, i) => {
        const x = pad + i * stepX;
        const y = yPx(Number(v || 0));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      // point markers
      ctx.fillStyle = color;
      arr.forEach((v, i) => {
        const x = pad + i * stepX;
        const y = yPx(Number(v || 0));
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    draw(seriesA, "#4fd6ff", "rgba(79, 214, 255, 0.7)");
    draw(seriesB, "#ff8b1a", "rgba(255, 139, 26, 0.6)");

    ctx.fillStyle = "#9ec3dd";
    ctx.font = "12px Segoe UI";
    ctx.fillText(`${labelA} (cyan)`, pad, 18);
    ctx.fillStyle = "#ffbb80";
    ctx.fillText(`${labelB} (orange)`, pad + 170, 18);

    ctx.fillStyle = "#6f90aa";
    ctx.fillText(`max ${yMaxRaw.toFixed(2)}`, 6, pad + 2);
    ctx.fillText(`min ${yMinRaw.toFixed(2)}`, 6, h - pad + 2);
    ctx.fillText("steps", w - 44, h - 10);
  }, [seriesA, seriesB, labelA, labelB]);

  return <canvas className="chart-canvas" ref={ref} width={1000} height={280} />;
}

export function CompareBars({ rows }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  return (
    <div className="compare-bars">
      {safeRows.map((row) => (
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
