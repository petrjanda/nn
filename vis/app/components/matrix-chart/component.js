import Ember from 'ember';
import layout from './template';

export default Ember.Component.extend({
  layout: layout,

  didInsertElement: function() {
    this.svg = d3.select(this.$()[0]).append('svg');

    this.svg
      .attr('width', 200)
      .attr('height', 1500)

    Ember.run.once(this, '_draw');
  },

  _draw: function() {
    var data = this.get('data');

    var ps = 12;
    var gapv = 0;
    var gaph = 0;

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
            var c = 255 - Math.abs(Math.round(d/2 * 255));
            var col = c.toString(16);

            console.log(col);

            return "#%@%@%@".fmt(col, col, col);
          })
          .on("mouseover", function(d, i) { 
            console.log(d) 
            self.set('current', [d, i]);
          })
  }

});
