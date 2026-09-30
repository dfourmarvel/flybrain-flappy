// Loads the live-mode model exported by src/flybrain/export_web.py (docs-site/live/).
// Works in the browser (fetch) and in Node tests (decodeCsr on bytes read with fs).

/** Decode the three little-endian CSR files (ArrayBuffers) into typed arrays. */
export function decodeCsr(model, indptrBuf, indicesBuf, weightsBuf) {
  const n = model.n_neurons;
  const nnz = model.nnz;
  const a = new DataView(indptrBuf);
  const b = new DataView(indicesBuf);
  const c = new DataView(weightsBuf);
  if (a.byteLength !== (n + 1) * 4) throw new Error(`csr_indptr.bin: expected ${(n + 1) * 4} bytes, got ${a.byteLength}`);
  if (b.byteLength !== nnz * 2) throw new Error(`csr_indices.bin: expected ${nnz * 2} bytes, got ${b.byteLength}`);
  if (c.byteLength !== nnz * 2) throw new Error(`csr_weights.bin: expected ${nnz * 2} bytes, got ${c.byteLength}`);
  // DataView with littleEndian=true, so decoding never depends on the host's byte order.
  const indptr = new Int32Array(n + 1);
  for (let i = 0; i <= n; i++) indptr[i] = a.getInt32(i * 4, true);
  const indices = new Uint16Array(nnz);
  const weights = new Int16Array(nnz);
  for (let p = 0; p < nnz; p++) {
    indices[p] = b.getUint16(p * 2, true);
    weights[p] = c.getInt16(p * 2, true);
  }
  if (indptr[0] !== 0 || indptr[n] !== nnz) throw new Error("csr_indptr.bin is inconsistent with nnz");
  return { indptr, indices, weights };
}

/**
 * Fetch model.json and the three .bin files from `baseUrl` (e.g. "live/").
 * @returns {Promise<{model: object, csr: {indptr: Int32Array, indices: Uint16Array, weights: Int16Array}}>}
 */
export async function loadLiveModel(baseUrl = "live/") {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const get = async (name) => {
    const res = await fetch(base + name);
    if (!res.ok) throw new Error(`failed to load ${base + name}: HTTP ${res.status}`);
    return res;
  };
  const model = await (await get("model.json")).json();
  const [ip, ix, w] = await Promise.all(
    ["csr_indptr.bin", "csr_indices.bin", "csr_weights.bin"].map(async (f) => (await get(f)).arrayBuffer())
  );
  return { model, csr: decodeCsr(model, ip, ix, w) };
}
