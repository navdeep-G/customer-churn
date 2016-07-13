
const fs = require('fs');
const jf = require('jsonfile')
const d3 = require('d3');
const _ = require('lodash');

const inputFile = 'telcoChurnTrainingSet0.750.csv';
const data = d3.csvParse(fs.readFileSync(inputFile, 'utf8'));
const outputData = [];

// get unique column names
const columnNames = _.keys(data[0]);
const ignore = ['customerID', 'Churn'];
const columnsToPlot = _.remove(columnNames, d => ignore.indexOf(d) === -1);

// console.log('columnNames', columnNames);
// console.log('columnsToPlot', columnsToPlot);

columnsToPlot.forEach(column => {
  console.log('isNaN(data[0][column])', isNaN(data[0][column]));

  if (isNaN(data[0][column])) {
    // calculate churn counts for our categorical features
    const type = 'categorical';
    const categoricalResponseCounts = [];
    const uniqueValues = _.uniq(data.map(row => row[column]));
    console.log(`uniqueValues for ${column}`, uniqueValues);

    uniqueValues.forEach(value => {
      const valueSubset = data.filter(d => d[column] === value);
      const churnedYesCount = valueSubset.filter(d => d.Churn === 'Yes').length;
      const churnedNoCount = valueSubset.filter(d => d.Churn === 'No').length;
      categoricalResponseCounts.push({
        value,
        churnedYesCount,
        churnedNoCount
      })
    })

    console.log('categoricalResponseCounts.length', categoricalResponseCounts.length);
    console.log('');
    outputData.push({
      column,
      type,
      responseCounts: categoricalResponseCounts
    })
  }

  if(!isNaN(data[0][column])) {
    // calculate churn counts for our numeric features
    const type = 'numeric';
    console.log('column', column);
    let histogram = d3.histogram()
      .value(d => Number(d[column]))
      .domain(d3.extent(data.map(e => Number(e[column]))))
      .thresholds(20);

    // console.log('typeof histogram(data)', typeof histogram(data));
    // console.log('_.keys(histogram(data))', _.keys(histogram(data)));
    const numericResponseCounts = [];

    const numericYesSubset = data.filter(d => d.Churn === 'Yes');
    const yesHistogramData = histogram(numericYesSubset);
    yesHistogramData.forEach((d, i) => {
      // console.log('_.keys(d)', _.keys(d));
      // console.log(`bin ${i} responseCount`, d.length);
      numericResponseCounts[i] = {};
      numericResponseCounts[i].churnedYesCount = d.length;
    });

    const numericNoSubset = data.filter(d => d.Churn === 'No');
    const noHistogramData = histogram(numericNoSubset);
    noHistogramData.forEach((d, i) => {
      // console.log(`bin ${i} responseCount`, d.length);
      numericResponseCounts[i].churnedNoCount = d.length;
      numericResponseCounts[i].x0 = d.x0;
      numericResponseCounts[i].x1 = d.x1;
    });

    console.log('numericResponseCounts.length', numericResponseCounts.length);
    console.log('');
    outputData.push({
      column,
      type,
      responseCounts: numericResponseCounts
    })
  }
})

const outputFile = 'response-distribution.json'
const outputJSONObj = outputData;

jf.writeFile(outputFile, outputJSONObj, {spaces: 2}, function(err){
  console.log(err)
})