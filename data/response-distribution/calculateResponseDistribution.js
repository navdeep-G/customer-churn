
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
let responseCounts;

columnsToPlot.forEach(column => {
  responseCounts = [];
  console.log('isNaN(data[0][column])', isNaN(data[0][column]));

  if (isNaN(data[0][column])) {
    // calculate churn counts for our categorical features
    const uniqueValues = _.uniq(data.map(row => row[column]));
    console.log(`uniqueValues for ${column}`, uniqueValues);

    uniqueValues.forEach(value => {
      const valueSubset = data.filter(d => d[column] === value);
      const churnedYesCount = valueSubset.filter(d => d.Churn === "Yes").length;
      const churnedNoCount = valueSubset.filter(d => d.Churn === "No").length;
      responseCounts.push({
        value,
        churnedYesCount,
        churnedNoCount
      })
    })
  }

  if(!isNaN(data[0][column])) {
    // calculate churn counts for our numeric features
    console.log('column', column);
    let histogram = d3.histogram()
      .value(d => Number(d[column]))
      .domain(d3.extent(data.map(e => Number(e[column]))))
      .thresholds(20);
    // console.log('typeof histogram(data)', typeof histogram(data));
    // console.log('_.keys(histogram(data))', _.keys(histogram(data)));
    _.keys(histogram(data)).forEach(key => {
      console.log('histogram(data)[key]', histogram(data)[key]);
      responseCounts.push(histogram(data)[key]);
    })
    // responseCounts.concat(histogram(data));
    responseCounts.forEach((d, i) => console.log(`bin ${i} responseCount`, d.length));
  }

  console.log('responseCounts.length', responseCounts.length);
  console.log('');
  outputData.push({
    column,
    responseCounts: responseCounts
  })
})

const outputFile = 'response-distribution.json'
const outputJSONObj = outputData;

jf.writeFile(outputFile, outputJSONObj, {spaces: 2}, function(err){
  console.log(err)
})