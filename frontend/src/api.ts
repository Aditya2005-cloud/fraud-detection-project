import type { PredictRequest, PredictResponse } from './types';

const API_BASE = '/api';

export async function predict(data: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || err.missing?.join(', ') || 'Prediction failed');
  }
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch('/api', { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}
