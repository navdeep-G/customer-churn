const jsonfile = require('jsonfile');
const d3 = require('d3');
const d3_request = require('d3-request');
d3.request = d3_request.request;

String.prototype.capitalizeFirstLetter = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}

// h2o-3 does not provide variable importance data for this model
const glmConfig = {
  server: 'localhost',
  port: '54321',
  type: 'Models',
  resourceId: 'glm-466538b6-2a41-4502-be46-463ffce2c616',
}

const gbmConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'gbm-bcef3445-8d00-4fb8-b823-8a86ce0ca77f',
  };

const drfConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'drf-f0a2ab1e-4079-4585-addc-c06698f05640',
  };

// h2o-3 does not provide variable importance data for this model
const dlConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'deeplearning-bc473fbe-5a25-4039-9ae4-e5a52c9ec73b',
  };

// h2o-3 does not provide variable importance data for this model
const nbConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'naivebayes-00e336fe-a8cc-46a7-bc60-c4dc1d6cf94e',
  };

const config = nbConfig;

const server = config.server;
const port = config.port;
const type = config.type;
const resourceId = config.resourceId;
const split = config.split;

const queryUrl = `http://${server}:${port}/3/${type}/${resourceId}`;

function parseAlgo(response) {
  const responseData = JSON.parse(response.responseText);
  const algo = responseData.models[0].algo;
  console.log('algo', algo);
  return algo;
}

function parseResponse(response) {
  const responseData = JSON.parse(response.responseText);
  console.log('responseData', responseData);
  const algo = responseData.models[0].algo;

  const metricsType = 'variable_importances';
  const variableImportanceData = responseData.models[0].output[metricsType].data;
  const columnData = variableImportanceData[0];
  const scaledImportanceData = variableImportanceData[2];

  const chartData = [];

  columnData.forEach(d => {
    chartData.push({
      'columnName': d
    })
  });

  scaledImportanceData.forEach((d, i) => {
    chartData[i].scaledImportance = d;
  })

  const parsedData = chartData;
  // console.log('parsedData', parsedData);
  return parsedData;
}

function callback(error, response) {
  // console.log('response', response);
  const outputData = parseResponse(response);
  const algo = parseAlgo(response);

  const outputFile = `${algo}-variable-importance-data.json`;
  jsonfile.spaces = 2;
  jsonfile.writeFile(outputFile, outputData, function (err) {
    console.error(err)
  })
}

d3.request(queryUrl).get(callback);