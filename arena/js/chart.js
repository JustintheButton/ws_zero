/* ws_zero arena: the line of the "who is winning" chart under a replayed game.

   values[i] is the estimate of the player who made move i, from that player's own side, or null if that player
   did not search (a random mover, say). The chart shows White's point of view, so Black's values change sign.
   The line joins every value there is: against an opponent that does not search every other value is missing, and
   lifting the pen at each gap would leave nothing drawn. */
(function (factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else globalThis.WSZ = Object.assign(globalThis.WSZ || {}, api);
})(function () {
  "use strict";

  /** The SVG path through the values: x(i) and y(value from White's side) give the coordinates. "" if no values. */
  function valuePath(values, whiteMovesFirst, x, y) {
    let d = "";
    values.forEach((v, i) => {
      if (v === null) return;
      const whiteMoved = (i % 2 === 0) === whiteMovesFirst;
      d += `${d ? "L" : "M"}${x(i).toFixed(1)},${y(whiteMoved ? v : -v).toFixed(1)}`;
    });
    return d;
  }

  return { valuePath };
});
