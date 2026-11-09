/* ws_zero arena: the network in TensorFlow.js (global `tf`), from a model exported by ws_zero.export: batch norm
   already folded into the convs, so every layer is conv + bias or linear + bias. Same layouts as ws_zero.net:
   observations (1, 8, 8, 119) NHWC, conv weights (kernel row, kernel column, in, out), "same" padding; the policy
   logit for observation square (row, column) and move type p is action (column * 8 + 7 - row) * 73 + p. */
(function (factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else globalThis.WSZ = Object.assign(globalThis.WSZ || {}, api);
})(function () {
  "use strict";
  const BLOCKS = 6;

  /** Build the network from a model's manifest ({arrays: [[name, shape, offset], ...]}) and its float32 data.
      Returns {evaluate(game) -> Promise<{logits: Float32Array(4672), value}>, dispose()}. */
  function loadNetwork(manifest, buffer) {
    const tf = globalThis.tf;
    const data = new Float32Array(buffer);
    const W = {};
    for (const [name, shape, offset] of manifest.arrays) {
      const size = shape.reduce((a, b) => a * b, 1);
      W[name] = tf.tensor(data.subarray(offset, offset + size), shape, "float32");
    }
    const conv = (x, name) => tf.add(tf.conv2d(x, W[`${name}/w`], 1, "same"), W[`${name}/b`]);

    function forward(obs) {
      return tf.tidy(() => {
        let x = tf.relu(conv(tf.tensor4d(obs, [1, 8, 8, 119]), "stem"));
        for (let i = 0; i < BLOCKS; i++) {
          const h = tf.relu(conv(x, `block${i}/conv1`));
          x = tf.relu(tf.add(x, conv(h, `block${i}/conv2`)));
        }
        const perSquare = conv(tf.relu(conv(x, "policy/conv")), "policy/out");  // (1, 8, 8, 73)
        const logits = tf.transpose(tf.reverse(perSquare, 1), [0, 2, 1, 3]).reshape([64 * 73]);
        const valueMap = tf.relu(conv(x, "value/conv")).reshape([1, 64]);
        const hidden = tf.relu(tf.add(tf.matMul(valueMap, W["value/fc1/w"]), W["value/fc1/b"]));
        const value = tf.tanh(tf.add(tf.matMul(hidden, W["value/fc2/w"]), W["value/fc2/b"])).reshape([1]);
        return [logits, value];
      });
    }

    async function evaluateObservation(obs) {
      const [l, v] = forward(obs);
      const [logits, value] = await Promise.all([l.data(), v.data()]);
      l.dispose();
      v.dispose();
      return { logits, value: value[0] };
    }

    return {
      evaluate: (game) => evaluateObservation(game.observation()),
      evaluateObservation,
      dispose: () => Object.values(W).forEach((t) => t.dispose()),
    };
  }

  return { loadNetwork };
});
