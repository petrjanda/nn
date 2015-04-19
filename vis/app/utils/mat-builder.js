export default function matBuilder(r, c, d) {
  var data = [];

  for(var i = 0; i < r; i++) {
    data[i] = d.slice(i*c, c*(i+1));
  }

  return {
    w: c,
    h : r,
    data: data
  }
}
