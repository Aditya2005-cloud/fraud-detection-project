import type { PredictRequest } from './types';

/** Random number in [min, max], 2 decimal places for amounts */
function rand(min: number, max: number, decimals = 0): number {
  const v = min + Math.random() * (max - min);
  return decimals ? Math.round(v * 100) / 100 : v;
}

/** Add small random noise to a value */
function noise(value: number, scale: number): number {
  return Math.round((value + rand(-scale, scale)) * 1000000) / 1000000;
}

const LEGIT_BASE: PredictRequest = {
  Time: 406,
  V1: -1.359807,
  V2: -0.072781,
  V3: 2.536347,
  V4: 1.378155,
  V5: -0.338321,
  V6: 0.462388,
  V7: 0.239599,
  V8: 0.098698,
  V9: 0.363787,
  V10: 0.090794,
  V11: -0.5516,
  V12: -0.617801,
  V13: -0.99139,
  V14: -0.311169,
  V15: 1.468177,
  V16: -0.470401,
  V17: 0.207971,
  V18: 0.025791,
  V19: 0.403993,
  V20: 0.251412,
  V21: -0.018307,
  V22: 0.277838,
  V23: -0.110474,
  V24: 0.066928,
  V25: 0.128539,
  V26: -0.189115,
  V27: 0.133558,
  V28: -0.021053,
  Amount: 149.62,
};

const FRAUD_BASE: PredictRequest = {
  Time: 119351,
  V1: -4.397974,
  V2: 1.358367,
  V3: -1.321837,
  V4: 2.469847,
  V5: 1.178755,
  V6: 0.135721,
  V7: 0.40696,
  V8: 0.167203,
  V9: -0.271149,
  V10: 0.716663,
  V11: -0.110474,
  V12: -0.617801,
  V13: -0.99139,
  V14: -0.311169,
  V15: 1.468177,
  V16: -0.470401,
  V17: 0.207971,
  V18: 0.025791,
  V19: 0.403993,
  V20: 0.251412,
  V21: -0.018307,
  V22: 0.277838,
  V23: -0.110474,
  V24: 0.066928,
  V25: 0.128539,
  V26: -0.189115,
  V27: 0.133558,
  V28: -0.021053,
  Amount: 2125.87,
};

/** Returns a new legitimate-style sample with random Time, Amount, and slightly varied V1–V28 each time */
export function getRandomLegitimateSample(): PredictRequest {
  const sample = { ...LEGIT_BASE };
  sample.Time = rand(0, 120000); // 0 to ~33 hours in sec
  sample.Amount = rand(1, 500, 2); // typical small–medium amounts
  const vKeys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'] as const;
  vKeys.forEach((k) => { sample[k] = noise(sample[k], 0.15); });
  return sample;
}

/** Returns a new fraud-style sample with random Time, Amount, and slightly varied V1–V28 each time */
export function getRandomFraudSample(): PredictRequest {
  const sample = { ...FRAUD_BASE };
  sample.Time = rand(50000, 150000); // different time range
  sample.Amount = rand(500, 3500, 2); // typically higher amounts
  const vKeys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'] as const;
  vKeys.forEach((k) => { sample[k] = noise(sample[k], 0.2); });
  return sample;
}

/** Single fixed sample for initial state (legitimate) */
export const sampleTransaction: PredictRequest = getRandomLegitimateSample();
