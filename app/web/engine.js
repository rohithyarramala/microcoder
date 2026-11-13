// JavaScript port of esp32/engine/microcoder.c (same blob format, same maths).
// Used by the browser demo so anyone can try the exact INT8 model that runs on the ESP32.
"use strict";
function MC(modelBuf, tokBuf) {
  const dv = new DataView(modelBuf);
  let off = 0;
  const i32 = () => { const v = dv.getInt32(off, true); off += 4; return v; };
  const pad = () => { off = (off + 3) & ~3; };
  if ((dv.getUint32(0, true) >>> 0) !== 0x3143434D) throw new Error("bad model blob");
  off = 4;
  const V = i32(), d = i32(), L = i32(), H = i32(), ffm = i32(), ctx = i32(); i32();
  const f = ffm * d, hd = d / H;
  const qmat = (rows, cols) => {
    const w = new Int8Array(modelBuf, off, rows * cols); off += rows * cols; pad();
    const s = new Float32Array(modelBuf.slice(off, off + rows * 4)); off += rows * 4; pad();
    return { w, s, rows, cols };
  };
  const vec = (n) => { const v = new Float32Array(modelBuf.slice(off, off + n * 4)); off += n * 4; return v; };
  const emb = qmat(V, d), nf = vec(d), layers = [];
  for (let l = 0; l < L; l++) {
    const n1 = vec(d), n2 = vec(d);
    layers.push({ n1, n2, qkv: qmat(3 * d, d), o: qmat(d, d), up: qmat(f, d), down: qmat(d, f) });
  }
  const K = new Float32Array(L * ctx * d), Vc = new Float32Array(L * ctx * d);
  const cosT = new Float32Array(ctx * hd / 2), sinT = new Float32Array(ctx * hd / 2);
  for (let t = 0; t < ctx; t++) for (let i = 0; i < hd / 2; i++) {
    const inv = 1 / Math.pow(10000, (2 * i) / hd);
    cosT[t * hd / 2 + i] = Math.cos(t * inv); sinT[t * hd / 2 + i] = Math.sin(t * inv);
  }
  const x = new Float32Array(d), xb = new Float32Array(d), xb2 = new Float32Array(d), hb = new Float32Array(f),
    q = new Float32Array(d), qkv = new Float32Array(3 * d), att = new Float32Array(ctx), logits = new Float32Array(V);
  const rms = (o, a, w) => { let ss = 0; for (let i = 0; i < d; i++) ss += a[i] * a[i]; ss = 1 / Math.sqrt(ss / d + 1e-5); for (let i = 0; i < d; i++) o[i] = a[i] * ss * w[i]; };
  const mv = (y, m, a) => { for (let r = 0; r < m.rows; r++) { let acc = 0; const b = r * m.cols; for (let c = 0; c < m.cols; c++) acc += m.w[b + c] * a[c]; y[r] = acc * m.s[r]; } };
  const rope = (arr, o, pos) => { for (let i = 0; i < hd / 2; i++) { const c = cosT[pos * hd / 2 + i], s = sinT[pos * hd / 2 + i], a = arr[o + 2 * i], b = arr[o + 2 * i + 1]; arr[o + 2 * i] = a * c - b * s; arr[o + 2 * i + 1] = a * s + b * c; } };
  const gelu = (v) => 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
  function forward(tok, pos) {
    for (let i = 0; i < d; i++) x[i] = emb.w[tok * d + i] * emb.s[tok];
    for (let l = 0; l < L; l++) {
      const Ly = layers[l], base = l * ctx * d;
      rms(xb, x, Ly.n1); mv(qkv, Ly.qkv, xb);
      for (let i = 0; i < d; i++) { q[i] = qkv[i]; K[base + pos * d + i] = qkv[d + i]; Vc[base + pos * d + i] = qkv[2 * d + i]; }
      for (let h = 0; h < H; h++) { rope(q, h * hd, pos); rope(K, base + pos * d + h * hd, pos); }
      const sc = 1 / Math.sqrt(hd);
      for (let h = 0; h < H; h++) {
        let mx = -1e30;
        for (let t = 0; t <= pos; t++) { let a = 0; const kb = base + t * d + h * hd; for (let i = 0; i < hd; i++) a += q[h * hd + i] * K[kb + i]; a *= sc; att[t] = a; if (a > mx) mx = a; }
        let sum = 0; for (let t = 0; t <= pos; t++) { att[t] = Math.exp(att[t] - mx); sum += att[t]; }
        for (let i = 0; i < hd; i++) xb2[h * hd + i] = 0;
        for (let t = 0; t <= pos; t++) { const w = att[t] / sum, vb = base + t * d + h * hd; for (let i = 0; i < hd; i++) xb2[h * hd + i] += w * Vc[vb + i]; }
      }
      mv(xb, Ly.o, xb2); for (let i = 0; i < d; i++) x[i] += xb[i];
      rms(xb, x, Ly.n2); mv(hb, Ly.up, xb); for (let i = 0; i < f; i++) hb[i] = gelu(hb[i]);
      mv(xb, Ly.down, hb); for (let i = 0; i < d; i++) x[i] += xb[i];
    }
    rms(xb, x, nf); mv(logits, emb, xb); return logits;
  }
  // ---- tokenizer (same algorithm as mc_tok_encode)
  const tdv = new DataView(tokBuf);
  let to = 0; const ti = () => { const v = tdv.getInt32(to, true); to += 4; return v; };
  if ((tdv.getUint32(0, true) >>> 0) !== 0x3154434D) throw new Error("bad tokenizer blob");
  to = 4; const TV = ti(), NM = ti(); const sp = { q: ti(), c: ti(), b: ti(), r: ti(), e: ti() };
  const offs = new Uint32Array(tokBuf.slice(to, to + (TV + 1) * 4)); to += (TV + 1) * 4;
  const bytes = new Uint8Array(tokBuf, to, offs[TV]); to += offs[TV]; to = (to + 3) & ~3;
  const merges = new Int32Array(tokBuf.slice(to, to + NM * 12));
  const rank = new Map(); for (let k = NM - 1; k >= 0; k--) rank.set(merges[3 * k] * 65536 + merges[3 * k + 1], k);
  const byteTok = new Int32Array(256).fill(-1);
  for (let i = 0; i < TV; i++) if (offs[i + 1] - offs[i] === 1 && byteTok[bytes[offs[i]]] < 0) byteTok[bytes[offs[i]]] = i;
  const isL = (c) => (c >= 97 && c <= 122) || (c >= 65 && c <= 90) || c >= 128, isN = (c) => c >= 48 && c <= 57,
    isS = (c) => c === 32 || (c >= 9 && c <= 13), isO = (c) => !isL(c) && !isN(c) && !isS(c);
  function pre(s, i) {
    const n = s.length;
    if (s[i] === 39 && i + 1 < n) for (const cs of ["s", "t", "re", "ve", "m", "ll", "d"]) {
      let ok = i + 1 + cs.length <= n; for (let k = 0; ok && k < cs.length; k++) ok = s[i + 1 + k] === cs.charCodeAt(k); if (ok) return i + 1 + cs.length;
    }
    const spc = s[i] === 32 && i + 1 < n;
    for (const P of [isL, isN, isO]) if ((spc && P(s[i + 1])) || P(s[i])) { let j = i + (P(s[i]) ? 0 : 1); while (j < n && P(s[j])) j++; return j; }
    let j = i; while (j < n && isS(s[j])) j++;
    return (j < n && j - i > 1) ? j - 1 : j;
  }
  function encode(text) {
    const s = new TextEncoder().encode(text), out = [];
    for (let i = 0; i < s.length;) {
      const j = pre(s, i); let w = []; for (let k = i; k < j; k++) w.push(byteTok[s[k]]);
      for (;;) {
        let best = -1, bi = -1;
        for (let k = 0; k + 1 < w.length; k++) { const r = rank.get(w[k] * 65536 + w[k + 1]); if (r !== undefined && (best < 0 || r < best)) { best = r; bi = k; } }
        if (best < 0) break; w.splice(bi, 2, merges[3 * best + 2]);
      }
      out.push(...w); i = j;
    }
    return out;
  }
  function encodePrompt(p) {
    const ids = []; const re = /<(q|c|b|r|e)>/g; let last = 0, m;
    while ((m = re.exec(p))) { ids.push(...encode(p.slice(last, m.index))); ids.push(sp[m[1]]); last = re.lastIndex; }
    ids.push(...encode(p.slice(last))); return ids;
  }
  const dec = new TextDecoder();
  function generate(prompt, maxNew = 120, temperature = 0) {
    const ids = encodePrompt(prompt); if (ids.length >= ctx - 8) return null;
    let lg; for (let p = 0; p < ids.length; p++) lg = forward(ids[p], p);
    const out = []; let pos = ids.length;
    while (out.length < maxNew && pos < ctx) {
      let best = 0; for (let i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
      if (temperature > 0) { let sum = 0; const pr = new Float64Array(V); for (let i = 0; i < V; i++) { pr[i] = Math.exp((lg[i] - lg[best]) / temperature); sum += pr[i]; } let r = Math.random() * sum; for (let i = 0; i < V; i++) { r -= pr[i]; if (r <= 0) { best = i; break; } } }
      if (best === sp.e) break; out.push(best); lg = forward(best, pos++);
    }
    const parts = out.map((id) => bytes.subarray(offs[id], offs[id + 1]));
    const all = new Uint8Array(parts.reduce((a, b) => a + b.length, 0)); let k = 0; for (const p of parts) { all.set(p, k); k += p.length; }
    return dec.decode(all);
  }
  return { generate, info: { V, d, L, H, ctx, params: V * d + L * (4 * d * d + 2 * f * d + 2 * d) + d, bytes: modelBuf.byteLength } };
}
