
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
  const responseCounts = [];

  // if the first observation of our column is not a number
  if (isNaN(data[0][column])) {
    const uniqueValues = _.uniq(data.map(row => row[column]));
    console.log('uniqueValues', uniqueValues);

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

  outputData.push({
    column,
    responseCounts
  })
})

const outputFile = 'response-distribution.json'
const outputJSONObj = outputData;

jf.writeFile(outputFile, outputJSONObj, {spaces: 2}, function(err){
  console.log(err)
})