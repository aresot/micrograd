import { Component } from '@angular/core';
import { Value } from './models/value';
import { MLP } from './models/nnet';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'micrograd';

  constructor() {
    // const a = new Value(2, 'a');
    // const b = new Value(-3, 'b');
    // const c = Value.mul(a, b, 'c');
    // const d = new Value(8, 'd');
    // const e = Value.add(c, d, 'e');
    // const f = Value.tanh(e, 'f');
    // f.backprop();

    // console.log(f.print());

    const mlp = new MLP(3, [4, 4, 1]);
    console.log(mlp);

    // const inp = [2, 3, 4].map(val => new Value(val, `inp`));
    // console.log('inp', inp.map(i => i.value));
    // const goal = new Value(-1, 'goal');

    // let loss = new Value(0);
    // for (let c = 0; c < 10; c++) {
    //   const res = mlp.call(inp);
    //   const delta = res[0].sub(goal);
    //   loss = delta.pow(2);
    //   console.log('loss', loss.value);
    //   loss.backprop();
    //   mlp.adjust(0.1);
    // }
    // mlp.print();
    // console.log(loss.print().join('\n'));

    const inputs = [
      [2, 3, -1],
      [3, -1, 0.5],
      [0.5, 1, 1],
      [1, 1, -1],
    ].map(vector => vector.map(val => new Value(val)));
    const ys = [1, -1, -1, 1].map(val => new Value(val));
    console.log('inputs', inputs);
    console.log('goal', ys.map(y => y.value));

    let loss = new Value(0);
    let step = 0.01;
    for (let c = 0; c < 1000; c++) {
      const outs = inputs.map(vector => {
        return mlp.call(vector)[0];
      });
      // console.log('outs', outs);

      loss = new Value(0);
      for (let i = 0; i < ys.length; i++) {
        const delta = outs[i].sub(ys[i]);
        loss = loss.add(delta.pow(2));
      }
      loss.backprop();
      mlp.adjust(step);
      step *= 0.999;
    }
    console.log(`loss=${loss.value} step=${step}`);
    const outs = inputs.map(vector => {
      return mlp.call(vector)[0];
    });
    console.log('outs', outs.map(out => out.value));

    // console.log(loss.print().join('\n'));
  }
}
