export type OperationType = '+' | '*' | 'tanh' | 'pow' | '-' | '';

export class Value {
  constructor(inp: number | Value, label = '') {
    this.value = inp instanceof Value ? inp.value : inp;
    this.label = label;
  }

  label = '';
  value: number;
  grad = 0;
  inputs: Value[] = [];
  operation: OperationType = '';
  _backprop: () => void = () => { };

  static add(inp1: Value, inp2: Value, label = ''): Value {
    const res = inp1.add(inp2);
    res.label = label;
    return res;
  }
  static sub(inp1: Value, inp2: Value, label = ''): Value {
    const res = inp1.sub(inp2);
    res.label = label;
    return res;
  }
  static mul(inp1: Value, inp2: Value, label = ''): Value {
    const res = inp1.mul(inp2);
    res.label = label;
    return res;
  }
  static tanh(inp: Value, label = ''): Value {
    const res = inp.tanh();
    res.label = label;
    return res;
  }

  add(inp: Value): Value {
    const res = new Value(this.value + inp.value);
    res.inputs = [this, inp];
    res.operation = '+';
    res._backprop = () => {
      res.inputs[0].grad += res.grad;
      res.inputs[1].grad += res.grad;
      res.inputs.forEach((input) => {
        input._backprop();
      });
    };
    return res;
  }
  sub(inp: Value): Value {
    const res = new Value(this.value - inp.value);
    res.inputs = [this, inp];
    res.operation = '-';
    res._backprop = () => {
      res.inputs[0].grad += res.grad;
      res.inputs[1].grad -= res.grad;
      res.inputs.forEach((input) => {
        input._backprop();
      });
    };
    return res;
  }
  mul(inp: Value): Value {
    const res = new Value(this.value * inp.value);
    res.inputs = [this, inp];
    res.operation = '*';
    res._backprop = () => {
      res.inputs[0].grad += res.inputs[1].value * res.grad;
      res.inputs[1].grad += res.inputs[0].value * res.grad;
      res.inputs.forEach((input) => {
        input._backprop();
      });
    };
    return res;
  }
  pow(power: number): Value {
    const res = new Value(Math.pow(this.value, power));
    res.inputs = [this];
    res.operation = 'pow';
    res._backprop = () => {
      res.inputs[0].grad += power * Math.pow(this.value, power - 1) * res.grad;
      res.inputs.forEach((input) => {
        input._backprop();
      });
    };
    return res;
  }
  tanh(): Value {
    const res = new Value(Math.tanh(this.value));
    res.inputs = [this];
    res.operation = 'tanh';
    res._backprop = () => {
      res.inputs[0].grad += (1 - res.value * res.value) * res.grad;
      res.inputs.forEach((input) => {
        input._backprop();
      });
    };
    return res;
  }

  backprop() {
    this.resetGrads();
    this.grad = 1;
    this._backprop();
  }

  print(lines: string[] = [], tabOffset = 0): string[] {
    const offset = Array(tabOffset * 3).fill(' ').join('');
    const op = this.operation ? `(${this.operation})` : '';
    const info = `${offset}${this.label}${op} | val=${this.value}, grad=${this.grad}`;
    lines.push(info);

    this.inputs.forEach((input) => {
      input.print(lines, tabOffset + 1);
    });

    return lines;
  }

  private resetGrads() {
    this.grad = 0;
    this.inputs.forEach((input) => {
      input.resetGrads();
    });
  }
}