/** Validation is side-effect free. Public batch writers finish validation and
 * allocation planning before committing any state or edge changes. */
export function finite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite, got ${value}`);
}
export function nonnegative(value: number, name: string): void {
  finite(value, name);
  if (value < 0) throw new Error(`${name} must be nonnegative, got ${value}`);
}
export function positive(value: number, name: string): void {
  finite(value, name);
  if (value <= 0) throw new Error(`${name} must be positive, got ${value}`);
}
export function integer(value: number, name: string, minimum = 0): void {
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}, got ${value}`);
}
export function neuronId(value: number, count: number): void {
  integer(value, "neuron id");
  if (value >= count) throw new Error(`neuron id out of range: ${value}`);
}
export function learning(repeats: number, eta: number, cap: number): void {
  integer(repeats, "repeats"); nonnegative(eta, "learning rate"); nonnegative(cap, "cap");
}
export function nonempty(values: Readonly<Record<string, unknown>>, name: string): void {
  if (!Object.keys(values).length) throw new Error(`${name} must contain evidence`);
}
