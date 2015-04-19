import Ember from 'ember';
import layout from './template';

export default Ember.Component.extend({
  layout: layout,

  didInsertElement: function() {
    this.svg = d3.select(this.$()[0]).append('svg');

    this.svg
      .attr('width', 900)
      .attr('height', 900)

    Ember.run.once(this, '_draw');
  },

  _draw: function() {
    var data = this.get('data');

    var ps = 12;
    var gapv = 1;
    var gaph = 1;

    var g = this.svg.selectAll('g')
      .data(data.data)
      .enter()
      .append('g')
      .attr("transform", function(d, i) { return "translate(0, " + i*(ps+gapv) + ")" })

    var self = this;

    var b = g.selectAll("rect")
      .data(function(d) { return d; })
      .enter()
        .append('rect')
          .attr('x', function(d, i) { return (ps+gaph)*i })        
          .attr('y', 0) 
          .attr('width', ps)
          .attr('height', ps)
          .attr('fill', function(d) {
            var c = 255 - Math.abs(Math.round(d * 255));
            var col = c.toString(16);

            console.log(col);

            return "#%@%@%@".fmt(col, col, col);
          })
          .on("mouseover", function(d, i) { 
            console.log(d) 
            self.set('current', [d, i]);
            // d3.select(this).attr('fill','red')
          })


//     for(var i = 0; i < data.w; i++) {
//       for(var j = 0; j < data.h; j++) {
//         this.svg.append('rect')
//           .attr('x', (ps+gaph)*i)        
//           .attr('y', (ps+gapv)*j)        
//           .attr('width', ps)
//           .attr('height', ps)
//           .attr('fill', function(d) {
//             var c = 255 - Math.abs(Math.round(data.data[i][j] * 255));
//             var col = c.toString(16);

//             console.log(col);

//             return "#%@%@%@".fmt(col, col, col);
//           })
//       }
    
//     }


  }

});
