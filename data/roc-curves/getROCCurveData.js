const jsonfile = require('jsonfile');
const d3 = require('d3');
const d3_request = require('d3-request');
d3.request = d3_request.request;

String.prototype.capitalizeFirstLetter = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}

const glmTrainConfig = {
  server: 'localhost',
  port: '54321',
  type: 'Models',
  resourceId: 'glm-466538b6-2a41-4502-be46-463ffce2c616',
  split: 'train'
}

const glmValidationConfig = {
  server: 'localhost',
  port: '54321',
  type: 'Models',
  resourceId: 'glm-466538b6-2a41-4502-be46-463ffce2c616',
  split: 'validation'
}

const gbmTrainConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'gbm-bcef3445-8d00-4fb8-b823-8a86ce0ca77f',
    split: 'train'
  };

const gbmValidationConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'gbm-bcef3445-8d00-4fb8-b823-8a86ce0ca77f',
    split: 'validation'
  };

const drfTrainConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'drf-f0a2ab1e-4079-4585-addc-c06698f05640',
    split: 'train'
  };

const drfValidationConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'drf-f0a2ab1e-4079-4585-addc-c06698f05640',
    split: 'validation'
  };

const dlTrainConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'deeplearning-bc473fbe-5a25-4039-9ae4-e5a52c9ec73b',
    split: 'train'
  };

const dlValidationConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'deeplearning-bc473fbe-5a25-4039-9ae4-e5a52c9ec73b',
    split: 'validation'
  };

const nbTrainConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'naivebayes-00e336fe-a8cc-46a7-bc60-c4dc1d6cf94e',
    split: 'train'
  };

const nbValidationConfig = {
    server: 'localhost',
    port: '54321',
    type: 'Models',
    resourceId: 'naivebayes-00e336fe-a8cc-46a7-bc60-c4dc1d6cf94e',
    split: 'validation'
  };

const config = nbValidationConfig;

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

  let metricsType;
  if (split === 'train') metricsType = 'training_metrics';
  if (split === 'validation') metricsType = 'validation_metrics';

  const rocData = responseData.models[0].output[metricsType].thresholds_and_metric_scores.data;
  const fprData = rocData[17];
  const tprData = rocData[18];

  const rocChartData = [];

  fprData.forEach(d => {
    rocChartData.push({
      'fpr': d
    })
  });

  //const tprKey = `${algo}${String(split).capitalizeFirstLetter()}`;
  tprData.forEach((d, i) => {
    rocChartData[i].tpr = d;
  })

  const parsedData = rocChartData;
  // console.log('parsedData', parsedData);
  return parsedData;
}

function callback(error, response) {
  // console.log('response', response);
  const outputData = parseResponse(response);
  const algo = parseAlgo(response);

  const outputFile = `${algo}-${split}-roc-data.json`;
  jsonfile.spaces = 2;
  jsonfile.writeFile(outputFile, outputData, function (err) {
    console.error(err)
  })
}

d3.request(queryUrl).get(callback);