// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Early stopping variables
const PATIENCE = 5; // Early stopping patience

// ========== DATA LOADING FUNCTIONS ==========

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        if (!trainData || trainData.length === 0) {
            throw new Error('Training data is empty or invalid');
        }
        
        if (!testData || testData.length === 0) {
            throw new Error('Test data is empty or invalid');
        }
        
        console.log('Training data loaded:', trainData.length, 'rows');
        console.log('Test data loaded:', testData.length, 'rows');
        console.log('First training row:', trainData[0]);
        
        statusDiv.innerHTML = `<p>Data loaded successfully!</p>
                              <p>Training: ${trainData.length} samples</p>
                              <p>Test: ${testData.length} samples</p>`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = 'Error loading data: ' + error.message;
        console.error('Load data error:', error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file: ' + e.target.error));
        reader.readAsText(file);
    });
}

// Improved CSV parser that handles quoted fields with commas correctly
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length < 2) return [];
    
    const headers = parseCSVLine(lines[0]);
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        
        // Ensure we have the right number of columns
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, idx) => {
                let val = values[idx];
                
                // Handle empty values
                if (val === '' || val === 'NULL' || val === 'null' || val === 'NaN' || val === undefined) {
                    row[header] = null;
                } 
                // Try to convert to number if possible
                else if (!isNaN(val) && val.trim() !== '') {
                    // Check if it's an integer or float
                    const numVal = Number(val);
                    // Only convert if it's actually a number (not NaN)
                    if (!isNaN(numVal)) {
                        row[header] = numVal;
                    } else {
                        row[header] = val;
                    }
                } 
                // Keep as string
                else {
                    row[header] = val;
                }
            });
            data.push(row);
        } else if (values.length > 0) {
            console.warn(`Skipping row ${i+1}: expected ${headers.length} columns, got ${values.length}`);
        }
    }
    
    return data;
}

// Parse a CSV line, handling quoted fields with commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            // Check if this is an escaped quote
            if (inQuotes && i < line.length - 1 && line[i + 1] === '"') {
                current += '"';
                i++; // Skip next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    result.push(current);
    
    return result;
}

// ========== DATA INSPECTION FUNCTIONS ==========

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = 'Dataset shape: ' + trainData.length + ' rows × ' + Object.keys(trainData[0]).length + ' columns';
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = 'Survival rate: ' + survivalCount + '/' + trainData.length + ' (' + survivalRate + '%)';
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => 
            row[feature] === null || row[feature] === undefined || 
            row[feature] === '' || (typeof row[feature] === 'number' && isNaN(row[feature]))
        ).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += '<li>' + feature + ': ' + missingPercent + '%</li>';
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += '<p>' + shapeInfo + '</p><p>' + targetInfo + '</p>' + missingInfo;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    if (!data || data.length === 0) {
        const emptyMsg = document.createElement('p');
        emptyMsg.textContent = 'No data available';
        return emptyMsg;
    }
    
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            if (value === null || value === undefined) {
                td.textContent = 'NULL';
                td.style.color = '#999';
            } else if (typeof value === 'number' && isNaN(value)) {
                td.textContent = 'NaN';
                td.style.color = '#999';
            } else {
                td.textContent = value;
            }
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    if (!trainData || trainData.length === 0) {
        chartsDiv.innerHTML += '<p>No data available for visualizations.</p>';
        return;
    }
    
    try {
        // Check if tfvis is available
        if (typeof tfvis === 'undefined') {
            chartsDiv.innerHTML += '<p>tfjs-vis is not loaded. Check if CDN is accessible.</p>';
            return;
        }
        
        // Show loading message
        chartsDiv.innerHTML += '<p>Loading charts in tfjs-vis visor (bottom right)...</p>';
        
        // Survival by Sex
        const survivalBySex = {};
        trainData.forEach(row => {
            if (row.Sex && row.Survived !== undefined && row.Survived !== null) {
                const sex = String(row.Sex).toLowerCase();
                if (!survivalBySex[sex]) {
                    survivalBySex[sex] = { survived: 0, total: 0 };
                }
                survivalBySex[sex].total++;
                if (row.Survived === 1) {
                    survivalBySex[sex].survived++;
                }
            }
        });
        
        // Prepare data for Survival by Sex
        const sexLabels = ['male', 'female'];
        const sexCounts = sexLabels.map(label => {
            const data = survivalBySex[label] || { survived: 0, total: 0 };
            return data.survived;
        });
        
        const sexRates = sexLabels.map(label => {
            const data = survivalBySex[label] || { survived: 0, total: 1 };
            return (data.survived / data.total) * 100;
        });
        
        // Plot 1: Survival Count by Sex - использовать правильный формат данных
        const sexCountData = [
            {index: 0, value: sexCounts[0]},
            {index: 1, value: sexCounts[1]}
        ];
        
        if (sexCountData.length > 0) {
            tfvis.render.barchart(
                { name: 'Survival Count by Sex', tab: 'Charts' },
                sexCountData,
                { 
                    xLabel: 'Sex',
                    yLabel: 'Count',
                    xTickLabels: ['Male', 'Female']
                }
            );
        }
        
        // Plot 2: Survival Rate by Sex (percentage)
        const sexRateData = [
            {index: 0, value: sexRates[0]},
            {index: 1, value: sexRates[1]}
        ];
        
        if (sexRateData.length > 0) {
            tfvis.render.barchart(
                { name: 'Survival Rate by Sex (%)', tab: 'Charts' },
                sexRateData,
                { 
                    xLabel: 'Sex',
                    yLabel: 'Percentage (%)',
                    xTickLabels: ['Male', 'Female']
                }
            );
        }
        
        // Survival by Pclass
        const survivalByPclass = {};
        trainData.forEach(row => {
            if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined && row.Survived !== null) {
                const pclass = String(row.Pclass);
                if (!survivalByPclass[pclass]) {
                    survivalByPclass[pclass] = { survived: 0, total: 0 };
                }
                survivalByPclass[pclass].total++;
                if (row.Survived === 1) {
                    survivalByPclass[pclass].survived++;
                }
            }
        });
        
        // Prepare data for Survival by Pclass - classes 1, 2, 3
        const pclassLabels = ['1', '2', '3'];
        const pclassCounts = pclassLabels.map(label => {
            const data = survivalByPclass[label] || { survived: 0, total: 0 };
            return data.survived;
        });
        
        const pclassRates = pclassLabels.map(label => {
            const data = survivalByPclass[label] || { survived: 0, total: 1 };
            return (data.survived / data.total) * 100;
        });
        
        // Plot 3: Survival Count by Pclass
        const pclassCountData = [
            {index: 0, value: pclassCounts[0]},
            {index: 1, value: pclassCounts[1]},
            {index: 2, value: pclassCounts[2]}
        ];
        
        if (pclassCountData.length > 0) {
            tfvis.render.barchart(
                { name: 'Survival Count by Passenger Class', tab: 'Charts' },
                pclassCountData,
                { 
                    xLabel: 'Passenger Class',
                    yLabel: 'Count',
                    xTickLabels: ['Class 1', 'Class 2', 'Class 3']
                }
            );
        }
        
        // Plot 4: Survival Rate by Pclass (percentage)
        const pclassRateData = [
            {index: 0, value: pclassRates[0]},
            {index: 1, value: pclassRates[1]},
            {index: 2, value: pclassRates[2]}
        ];
        
        if (pclassRateData.length > 0) {
            tfvis.render.barchart(
                { name: 'Survival Rate by Passenger Class (%)', tab: 'Charts' },
                pclassRateData,
                { 
                    xLabel: 'Passenger Class',
                    yLabel: 'Percentage (%)',
                    xTickLabels: ['Class 1', 'Class 2', 'Class 3']
                }
            );
        }
        
        // Age distribution - filter invalid ages
        const ageValues = trainData
            .map(row => row.Age)
            .filter(age => age !== null && age !== undefined && 
                    !isNaN(age) && typeof age === 'number' && 
                    age >= 1 && age <= 100);
        
        if (ageValues.length > 0) {
            const ageData = ageValues.map(age => ({ value: age }));
            
            tfvis.render.histogram(
                { name: 'Age Distribution', tab: 'Charts' },
                ageData,
                { 
                    xLabel: 'Age (years)', 
                    yLabel: 'Count',
                    maxBins: 20
                }
            );
        }
        
        // Fare distribution - filter invalid fares
        const fareValues = trainData
            .map(row => row.Fare)
            .filter(fare => fare !== null && fare !== undefined && 
                    !isNaN(fare) && typeof fare === 'number' && 
                    fare >= 1 && fare <= 300);
        
        if (fareValues.length > 0) {
            const fareData = fareValues.map(fare => ({ value: fare }));
            
            tfvis.render.histogram(
                { name: 'Fare Distribution (USD)', tab: 'Charts' },
                fareData,
                { 
                    xLabel: 'Fare (USD)', 
                    yLabel: 'Count',
                    maxBins: 20
                }
            );
        }
        
        // Add summary text
        const maleData = survivalBySex['male'] || { survived: 0, total: 0 };
        const femaleData = survivalBySex['female'] || { survived: 0, total: 0 };
        const class1Data = survivalByPclass['1'] || { survived: 0, total: 0 };
        const class2Data = survivalByPclass['2'] || { survived: 0, total: 0 };
        const class3Data = survivalByPclass['3'] || { survived: 0, total: 0 };
        
        chartsDiv.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background-color: #f0f7ff; border-radius: 5px;">
                <h4>Chart Summary:</h4>
                <ul>
                    <li>Male survival: ${maleData.survived}/${maleData.total} (${maleData.total > 0 ? ((maleData.survived / maleData.total) * 100).toFixed(1) : 0}%)</li>
                    <li>Female survival: ${femaleData.survived}/${femaleData.total} (${femaleData.total > 0 ? ((femaleData.survived / femaleData.total) * 100).toFixed(1) : 0}%)</li>
                    <li>Class 1 survival: ${class1Data.survived}/${class1Data.total} (${class1Data.total > 0 ? ((class1Data.survived / class1Data.total) * 100).toFixed(1) : 0}%)</li>
                    <li>Class 2 survival: ${class2Data.survived}/${class2Data.total} (${class2Data.total > 0 ? ((class2Data.survived / class2Data.total) * 100).toFixed(1) : 0}%)</li>
                    <li>Class 3 survival: ${class3Data.survived}/${class3Data.total} (${class3Data.total > 0 ? ((class3Data.survived / class3Data.total) * 100).toFixed(1) : 0}%)</li>
                </ul>
                <p>Charts loaded successfully! Click the tfjs-vis button (bottom right) to view all charts.</p>
            </div>
        `;
        
    } catch (error) {
        console.error('Error creating visualizations:', error);
        chartsDiv.innerHTML += '<p>Error creating visualizations: ' + error.message + '</p>';
    }
}

// ========== DATA PREPROCESSING FUNCTIONS ==========

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation values from training data
        const ageValues = trainData.map(row => row.Age)
            .filter(age => age !== null && age !== undefined && 
                    !isNaN(age) && typeof age === 'number' && 
                    age >= 1 && age <= 100);
        
        const fareValues = trainData.map(row => row.Fare)
            .filter(fare => fare !== null && fare !== undefined && 
                    !isNaN(fare) && typeof fare === 'number' && 
                    fare >= 1);
        
        const embarkedValues = trainData.map(row => row.Embarked)
            .filter(e => e !== null && e !== undefined && e !== '' && String(e).trim() !== '');
        
        const ageMedian = ageValues.length > 0 ? calculateMedian(ageValues) : 30;
        const fareMedian = fareValues.length > 0 ? calculateMedian(fareValues) : 32;
        const embarkedMode = embarkedValues.length > 0 ? calculateMode(embarkedValues) : 'S';
        
        console.log('Imputation values:', { ageMedian, fareMedian, embarkedMode });
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            if (features && features.length > 0) {
                preprocessedTrainData.features.push(features);
                const label = row[TARGET_FEATURE];
                // Ensure label is 0 or 1
                const labelValue = label !== null && label !== undefined ? Number(label) : 0;
                preprocessedTrainData.labels.push(labelValue === 1 ? 1 : 0);
            }
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            if (features && features.length > 0) {
                preprocessedTestData.features.push(features);
                const passengerId = row[ID_FEATURE];
                preprocessedTestData.passengerIds.push(
                    passengerId !== null && passengerId !== undefined ? String(passengerId) : ''
                );
            }
        });
        
        // Check if we have data
        if (preprocessedTrainData.features.length === 0) {
            throw new Error('No valid features extracted from training data');
        }
        
        if (preprocessedTestData.features.length === 0) {
            throw new Error('No valid features extracted from test data');
        }
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        const featureCount = preprocessedTrainData.features.shape[1];
        const sampleCount = preprocessedTrainData.features.shape[0];
        
        outputDiv.innerHTML = 
            '<p>Preprocessing completed!</p>' +
            '<p>Training samples: ' + sampleCount + '</p>' +
            '<p>Feature count: ' + featureCount + '</p>' +
            '<p>Imputation values: </p>' +
            '<ul>' +
                '<li>Age median: ' + ageMedian.toFixed(2) + ' years</li>' +
                '<li>Fare median: ' + fareMedian.toFixed(2) + ' USD</li>' +
                '<li>Embarked mode: ' + embarkedMode + '</li>' +
            '</ul>';
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = 'Error during preprocessing: ' + error.message;
        console.error('Preprocessing error:', error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    try {
        // Impute missing values with validation
        const age = (row.Age !== null && row.Age !== undefined && 
                    !isNaN(row.Age) && typeof row.Age === 'number' && 
                    row.Age >= 1 && row.Age <= 100) 
            ? row.Age : ageMedian;
        
        const fare = (row.Fare !== null && row.Fare !== undefined && 
                     !isNaN(row.Fare) && typeof row.Fare === 'number' && 
                     row.Fare >= 1) 
            ? row.Fare : fareMedian;
        
        const embarked = (row.Embarked !== null && row.Embarked !== undefined && 
                         row.Embarked !== '' && String(row.Embarked).trim() !== '') 
            ? String(row.Embarked).trim() : embarkedMode;
        
        // Standardize numerical features - use only valid values
        const ageValues = trainData.map(r => r.Age)
            .filter(a => a !== null && a !== undefined && 
                    !isNaN(a) && typeof a === 'number' && 
                    a >= 1 && a <= 100);
        const fareValues = trainData.map(r => r.Fare)
            .filter(f => f !== null && f !== undefined && 
                    !isNaN(f) && typeof f === 'number' && 
                    f >= 1);
        
        const ageStdDev = calculateStdDev(ageValues) || 1;
        const fareStdDev = calculateStdDev(fareValues) || 1;
        
        const standardizedAge = (age - ageMedian) / ageStdDev;
        const standardizedFare = (fare - fareMedian) / fareStdDev;
        
        // One-hot encode categorical features
        const pclass = (row.Pclass !== null && row.Pclass !== undefined && 
                       [1, 2, 3].includes(Number(row.Pclass))) 
            ? Number(row.Pclass) : 3;
        const sex = (row.Sex !== null && row.Sex !== undefined && 
                    String(row.Sex).trim() !== '') 
            ? String(row.Sex).trim().toLowerCase() : 'male';
        
        const pclassOneHot = oneHotEncode(pclass, [1, 2, 3]);
        const sexOneHot = oneHotEncode(sex, ['male', 'female']);
        const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
        
        // Start with numerical features
        let features = [
            standardizedAge,
            standardizedFare,
            (row.SibSp !== null && row.SibSp !== undefined ? Number(row.SibSp) : 0),
            (row.Parch !== null && row.Parch !== undefined ? Number(row.Parch) : 0)
        ];
        
        // Add one-hot encoded features
        features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
        
        // Add optional family features if enabled
        if (document.getElementById('add-family-features').checked) {
            const sibSp = row.SibSp !== null && row.SibSp !== undefined ? Number(row.SibSp) : 0;
            const parch = row.Parch !== null && row.Parch !== undefined ? Number(row.Parch) : 0;
            const familySize = sibSp + parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            features.push(familySize, isAlone);
        }
        
        return features;
    } catch (error) {
        console.error('Error extracting features:', error, row);
        return null;
    }
}

// Calculate median of an array
function calculateMedian(values) {
    if (!values || values.length === 0) return 0;
    
    const validValues = values.filter(v => v !== null && v !== undefined && !isNaN(v) && typeof v === 'number');
    if (validValues.length === 0) return 0;
    
    const sorted = [...validValues].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
        return (sorted[half - 1] + sorted[half]) / 2;
    }
    
    return sorted[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (!values || values.length === 0) return null;
    
    const frequency = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
        const val = String(value).trim();
        frequency[val] = (frequency[val] || 0) + 1;
        if (frequency[val] > maxCount) {
            maxCount = frequency[val];
            mode = val;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (!values || values.length === 0) return 0;
    
    const validValues = values.filter(v => v !== null && v !== undefined && !isNaN(v) && typeof v === 'number');
    if (validValues.length === 0) return 0;
    
    const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
    const squaredDiffs = validValues.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / validValues.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    } else if (categories.length > 0) {
        // Default to first category if value not found
        encoding[0] = 1;
    }
    return encoding;
}

// ========== MODEL CREATION FUNCTIONS ==========

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create a sequential model
    model = tf.sequential();
    
    // Add layers - single hidden layer with sigmoid gate
    model.add(tf.layers.dense({
        units: 16,
        activation: 'sigmoid', // Changed from 'relu' to 'sigmoid' for feature importance interpretation
        inputShape: [inputShape],
        kernelInitializer: 'glorotNormal',
        name: 'hidden_layer'
    }));
    
    // Output layer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output_layer'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.name || layer.getClassName()} 
                      - Units: ${layer.units || 'N/A'} 
                      - Activation: ${layer.activation ? layer.activation.getClassName() : 'N/A'} 
                      - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams().toLocaleString()}</p>`;
    summaryText += '<p>Compiled with: optimizer=adam, loss=binaryCrossentropy, metrics=[accuracy]</p>';
    summaryText += '<p>Note: Using sigmoid activation in hidden layer allows for better interpretation of feature importance.</p>';
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// ========== TRAINING FUNCTIONS ==========

// Custom callback for early stopping
class EarlyStoppingCallback {
    constructor(patience = 5) {
        this.patience = patience;
        this.bestValLoss = Infinity;
        this.wait = 0;
        this.stoppedEpoch = 0;
    }

    onEpochEnd(epoch, logs) {
        const valLoss = logs.val_loss;
        
        if (valLoss < this.bestValLoss) {
            this.bestValLoss = valLoss;
            this.wait = 0;
        } else {
            this.wait++;
            if (this.wait >= this.patience) {
                this.model.stopTraining = true;
                this.stoppedEpoch = epoch;
                console.log('Early stopping triggered at epoch ' + (epoch + 1));
            }
        }
    }

    setModel(model) {
        this.model = model;
    }
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...<br>';
    
    try {
        // Split training data into train and validation sets (80/20 split)
        const totalSamples = preprocessedTrainData.features.shape[0];
        const splitIndex = Math.floor(totalSamples * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Create callbacks
        const earlyStopping = new EarlyStoppingCallback(PATIENCE);
        
        const callbacks = [
            tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                { height: 300, width: 500, callbacks: ['onEpochEnd'] }
            ),
            earlyStopping
        ];
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: callbacks,
            verbose: 1,
            yieldEvery: 'epoch'
        });
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Check if early stopping was triggered
        if (trainingHistory.epoch.length < 50) {
            statusDiv.innerHTML += `<p>Early stopping triggered at epoch ${trainingHistory.epoch.length} (patience=${PATIENCE})</p>`;
        } else {
            statusDiv.innerHTML += '<p>Training completed all 50 epochs.</p>';
        }
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
        
        // Visualize feature importance
        visualizeFeatureImportance();
    } catch (error) {
        statusDiv.innerHTML = 'Error during training: ' + error.message;
        console.error('Training error:', error);
    }
}

// ========== EVALUATION FUNCTIONS ==========

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) {
        console.log('No validation data available');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    try {
        // Calculate confusion matrix
        const predVals = await validationPredictions.array();
        const trueVals = await validationLabels.array();
        
        // Ensure arrays are flat
        const flatPredVals = Array.isArray(predVals[0]) ? predVals.map(p => p[0]) : predVals;
        const flatTrueVals = Array.isArray(trueVals[0]) ? trueVals.map(t => t[0]) : trueVals;
        
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (let i = 0; i < flatPredVals.length; i++) {
            const prediction = flatPredVals[i] >= threshold ? 1 : 0;
            const actual = flatTrueVals[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        // Update confusion matrix display
        const cmDiv = document.getElementById('confusion-matrix');
        cmDiv.innerHTML = 
            `<div style="overflow-x: auto;">
                <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"></th>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Predicted Positive</th>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Predicted Negative</th>
                    </tr>
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Actual Positive</th>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: ${tp > 0 ? '#d4edda' : 'white'};">${tp}</td>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: ${fn > 0 ? '#f8d7da' : 'white'};">${fn}</td>
                    </tr>
                    <tr>
                        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Actual Negative</th>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: ${fp > 0 ? '#f8d7da' : 'white'};">${fp}</td>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: ${tn > 0 ? '#d4edda' : 'white'};">${tn}</td>
                    </tr>
                </table>
            </div>`;
        
        // Calculate performance metrics
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
        
        // Update performance metrics display
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = 
            `<p><strong>Accuracy:</strong> <span style="color: #1a73e8;">${(accuracy * 100).toFixed(2)}%</span></p>
             <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
             <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
             <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>`;
        
        // Calculate and plot ROC curve
        await plotROC(flatTrueVals, flatPredVals);
        
        // Calculate and display AUC
        const auc = calculateAUC(flatTrueVals, flatPredVals);
        metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
        
        // Display threshold info
        metricsDiv.innerHTML += `<p><em>Using threshold: ${threshold.toFixed(2)}</em></p>`;
    } catch (error) {
        console.error('Error updating metrics:', error);
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = `<p>Error calculating metrics: ${error.message}</p>`;
    }
}

// Calculate AUC using trapezoidal rule
function calculateAUC(trueLabels, predictions) {
    // Sort by prediction score descending
    const combined = trueLabels.map((label, i) => ({
        label: label,
        score: predictions[i]
    })).sort((a, b) => b.score - a.score);
    
    // Calculate ROC points
    let tpr = 0;
    let fpr = 0;
    let prevTpr = 0;
    let prevFpr = 0;
    let auc = 0;
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    const totalPositives = trueLabels.filter(l => l === 1).length;
    const totalNegatives = trueLabels.filter(l => l === 0).length;
    
    for (const item of combined) {
        if (item.label === 1) {
            tp++;
            fn = totalPositives - tp;
        } else {
            fp++;
            tn = totalNegatives - fp;
        }
        
        prevTpr = tpr;
        prevFpr = fpr;
        tpr = tp / totalPositives;
        fpr = fp / totalNegatives;
        
        // Add area of trapezoid
        auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
    }
    
    return auc;
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    try {
        // Calculate TPR and FPR for different thresholds
        const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
        const rocData = [];
        
        const totalPositives = trueLabels.filter(l => l === 1).length;
        const totalNegatives = trueLabels.filter(l => l === 0).length;
        
        thresholds.forEach(threshold => {
            let tp = 0, fp = 0;
            
            for (let i = 0; i < predictions.length; i++) {
                const prediction = predictions[i] >= threshold ? 1 : 0;
                const actual = trueLabels[i];
                
                if (prediction === 1 && actual === 1) tp++;
                else if (prediction === 1 && actual === 0) fp++;
            }
            
            const tpr = tp / totalPositives || 0;
            const fpr = fp / totalNegatives || 0;
            
            rocData.push({ threshold: threshold, fpr: fpr, tpr: tpr });
        });
        
        // Plot ROC curve if tfvis is available
        if (typeof tfvis !== 'undefined') {
            const rocPoints = rocData.map(d => ({ x: d.fpr, y: d.tpr }));
            
            tfvis.render.linechart(
                { name: 'ROC Curve', tab: 'Evaluation' },
                {
                    values: rocPoints,
                    series: ['ROC Curve']
                },
                { 
                    xLabel: 'False Positive Rate', 
                    yLabel: 'True Positive Rate',
                    width: 450,
                    height: 400,
                    xAxisDomain: [0, 1],
                    yAxisDomain: [0, 1]
                }
            );
        }
    } catch (error) {
        console.error('Error plotting ROC curve:', error);
    }
}

// ========== PREDICTION FUNCTIONS ==========

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = await testPredictions.array();
        
        // Ensure predictions are flat array
        const flatPredValues = Array.isArray(predValues[0]) ? predValues.map(p => p[0]) : predValues;
        
        // Create prediction results with validation
        const results = [];
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const passengerId = preprocessedTestData.passengerIds[i];
            const probability = flatPredValues[i];
            
            if (passengerId && probability !== undefined && probability !== null && !isNaN(probability)) {
                results.push({
                    PassengerId: passengerId,
                    Survived: probability >= 0.5 ? 1 : 0,
                    Probability: probability
                });
            }
        }
        
        if (results.length === 0) {
            outputDiv.innerHTML = '<p>No valid predictions generated. Check your data.</p>';
            return;
        }
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        outputDiv.innerHTML += `<p>Predictions completed! Total valid predictions: ${results.length} samples</p>`;
        outputDiv.innerHTML += `<p>Prediction threshold: 0.5</p>`;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = 'Error during prediction: ' + error.message;
        console.error('Prediction error:', error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    table.style.margin = '10px 0';
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.textAlign = 'left';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        
        // PassengerId
        const tdId = document.createElement('td');
        tdId.textContent = row.PassengerId || '';
        tdId.style.border = '1px solid #ddd';
        tdId.style.padding = '8px';
        tr.appendChild(tdId);
        
        // Survived
        const tdSurvived = document.createElement('td');
        const survivedValue = row.Survived !== null && row.Survived !== undefined ? row.Survived : '';
        tdSurvived.textContent = survivedValue;
        tdSurvived.style.border = '1px solid #ddd';
        tdSurvived.style.padding = '8px';
        tdSurvived.style.color = survivedValue === 1 ? '#28a745' : '#dc3545';
        tdSurvived.style.fontWeight = 'bold';
        tr.appendChild(tdSurvived);
        
        // Probability
        const tdProb = document.createElement('td');
        const probValue = row.Probability;
        if (probValue !== null && probValue !== undefined && typeof probValue === 'number' && !isNaN(probValue)) {
            tdProb.textContent = probValue.toFixed(4);
            // Color code based on probability
            if (probValue >= 0.5) {
                tdProb.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
            } else {
                tdProb.style.backgroundColor = 'rgba(220, 53, 69, 0.1)';
            }
        } else {
            tdProb.textContent = 'N/A';
        }
        tdProb.style.border = '1px solid #ddd';
        tdProb.style.padding = '8px';
        tr.appendChild(tdProb);
        
        table.appendChild(tr);
    });
    
    return table;
}

// ========== EXPORT FUNCTIONS ==========

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Get predictions
        const predValues = await testPredictions.array();
        
        // Ensure predictions are flat array
        const flatPredValues = Array.isArray(predValues[0]) ? predValues.map(p => p[0]) : predValues;
        
        // Create submission CSV (PassengerId, Survived) - Kaggle format
        let submissionCSV = 'PassengerId,Survived\n';
        let probabilitiesCSV = 'PassengerId,Probability\n';
        
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const passengerId = preprocessedTestData.passengerIds[i];
            const probability = flatPredValues[i];
            
            if (passengerId && probability !== undefined && probability !== null && !isNaN(probability)) {
                submissionCSV += `${passengerId},${probability >= 0.5 ? 1 : 0}\n`;
                probabilitiesCSV += `${passengerId},${probability.toFixed(6)}\n`;
            }
        }
        
        // Create download links
        const downloadFile = (content, filename) => {
            const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };
        
        // Download files
        downloadFile(submissionCSV, 'submission.csv');
        downloadFile(probabilitiesCSV, 'probabilities.csv');
        
        // Save model
        await model.save('downloads://titanic-tfjs-model');
        
        statusDiv.innerHTML = 
            '<p>Export completed!</p>' +
            '<p>Downloaded files:</p>' +
            '<ul>' +
                '<li><strong>submission.csv</strong> - Kaggle submission format (PassengerId, Survived)</li>' +
                '<li><strong>probabilities.csv</strong> - Prediction probabilities for each passenger</li>' +
            '</ul>' +
            '<p>Model saved to browser downloads as "titanic-tfjs-model"</p>' +
            '<p>Note: The model files (.json and .bin) have been downloaded to your default download folder.</p>';
    } catch (error) {
        statusDiv.innerHTML = 'Error during export: ' + error.message;
        console.error('Export error:', error);
    }
}

// ========== FEATURE IMPORTANCE FUNCTIONS ==========

// Sigmoid gate visualization and feature importance
function visualizeFeatureImportance() {
    if (!model || !preprocessedTrainData) {
        console.log('Model or training data not available');
        return;
    }
    
    try {
        // Get the hidden layer weights
        const hiddenLayer = model.getLayer('hidden_layer');
        if (!hiddenLayer) return;
        
        const weights = hiddenLayer.getWeights()[0]; // Get kernel weights
        const weightsArray = weights.arraySync();
        
        // Calculate feature importance (average absolute weight per input feature)
        const featureImportance = [];
        const numFeatures = weightsArray.length;
        const numNeurons = weightsArray[0].length;
        
        for (let i = 0; i < numFeatures; i++) {
            let sumAbsWeights = 0;
            for (let j = 0; j < numNeurons; j++) {
                sumAbsWeights += Math.abs(weightsArray[i][j]);
            }
            featureImportance.push({
                featureIndex: i,
                importance: sumAbsWeights / numNeurons
            });
        }
        
        // Sort by importance
        featureImportance.sort((a, b) => b.importance - a.importance);
        
        // Create feature names based on our preprocessing order
        const featureNames = [
            'Age (standardized)', 
            'Fare (standardized)', 
            'SibSp', 
            'Parch',
            'Pclass=1', 'Pclass=2', 'Pclass=3',
            'Sex=male', 'Sex=female',
            'Embarked=C', 'Embarked=Q', 'Embarked=S'
        ];
        
        // Add family features if enabled
        if (document.getElementById('add-family-features').checked) {
            featureNames.push('FamilySize', 'IsAlone');
        }
        
        // Prepare data for visualization
        const importanceData = featureImportance.map(item => ({
            feature: featureNames[item.featureIndex] || `Feature ${item.featureIndex}`,
            importance: item.importance
        }));
        
        // Display feature importance
        if (typeof tfvis !== 'undefined' && importanceData.length > 0) {
            const chartData = importanceData.map((d, index) => ({
                index: index,
                value: d.importance
            }));
            
            tfvis.render.barchart(
                { name: 'Feature Importance (Sigmoid Gate)', tab: 'Analysis' },
                chartData,
                { 
                    xLabel: 'Feature', 
                    yLabel: 'Average Absolute Weight',
                    width: 500,
                    height: 400
                }
            );
            
            console.log('Feature importance calculated (using sigmoid activation):', importanceData);
        }
    } catch (error) {
        console.error('Error visualizing feature importance:', error);
    }
}

// ========== GLOBAL EXPORTS ==========
// Make functions available globally for HTML onclick handlers
window.loadData = loadData;
window.inspectData = inspectData;
window.preprocessData = preprocessData;
window.createModel = createModel;
window.trainModel = trainModel;
window.predict = predict;
window.exportResults = exportResults;
window.updateMetrics = updateMetrics;
window.visualizeFeatureImportance = visualizeFeatureImportance;