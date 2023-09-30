import { Value } from "./value";

const mulberry32 = (a: number) => {
  return function () {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
};
const rand = mulberry32(13423);

export class Neuron {
  constructor(size: number, label = '') {
    this.bias = new Value(0, `${label}b`);
    this.weights = [];
    for (let i = 0; i < size; i++) {
      this.weights.push(new Value(rand() * 2 - 1, `${label}w${i}`));
    }
  }

  weights: Value[];
  bias: Value;

  call(values: Value[]): Value {
    let sum = new Value(this.bias.value, this.bias.label);
    for (let i = 0; i < values.length; i++) {
      sum = sum.add(this.weights[i].mul(values[i]));
    }

    return sum.tanh();
  }

  adjust(step: number) {
    this.bias.value -= step * this.bias.grad;
    this.weights.forEach((weight) => {
      weight.value -= step * weight.grad;
    });
  }

  print(): string {
    const w = this.weights.map(w => `${w.value}|${w.grad}`);
    return `{${this.bias.value}|${this.bias.grad}, ${w.join(', ')}}`;
  }
}

export class Layer {
  constructor(nInputs: number, total: number, label = '') {
    this.neurons = [];
    for (let i = 0; i < total; i++) {
      this.neurons.push(new Neuron(nInputs, `${label}N${i}`));
    }
  }

  neurons: Neuron[];

  call(input: Value[]): Value[] {
    const res: Value[] = [];

    this.neurons.forEach((neuron) => {
      res.push(neuron.call(input));
    });

    return res;
  }

  adjust(step: number) {
    this.neurons.forEach((neuron) => {
      neuron.adjust(step);
    });
  }

  print(): string {
    return `[${this.neurons.map(n => n.print()).join(', ')}]`;
  }
}

export class MLP {
  constructor(nInputes: number, layerSizes: number[]) {
    this.layers = [];

    for (let i = 0; i < layerSizes.length; i++) {
      this.layers.push(new Layer(i === 0 ? nInputes : layerSizes[i - 1], layerSizes[i], `L${i}`));
    }
  }

  layers: Layer[];

  call(input: Value[]): Value[] {
    let res = input;
    this.layers.forEach((layer) => {
      res = layer.call(res);
    });

    return res;
  }

  adjust(step: number) {
    this.layers.forEach((layer) => {
      layer.adjust(step);
    });
  }

  print() {
    const layers = this.layers.map(layer => layer.print());
    console.log(`MLP:\n${layers.join('\n')}`);
  }
}