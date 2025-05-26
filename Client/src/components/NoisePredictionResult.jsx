import React from "react";

const NoisePredictionResult = ({ spectrogramUrl, prediction, confidence, topPredictions, handleButtonClick }) => {
  return (
    <section className="w-full flex justify-center bg-blue-50 mb">
      <div className="w-4/5 mt-10 p-6 bg-white rounded-xl shadow-xl md:w-2/4">
        {/* Heading */}
        <h2 className="text-3xl font-bold text-blue-800 text-center mb-8">
          Noise Classification Result
        </h2>

        <div className="grid md:grid-cols-2 gap-10 items-start">
          {/* Spectrogram Image */}
          <div className="w-full h-70 bg-gray-100 rounded-lg overflow-hidden shadow-inner">
            {spectrogramUrl ? (
              
              <img src={`data:image/png;base64,${spectrogramUrl}`} 
                alt="Mel-Spectrogram"
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-gray-400 font-medium flex items-center justify-center h-full">
                No spectrogram uploaded
              </span>
            )}
          </div>

          {/* Prediction Result */}
          <div className="bg-blue-50 rounded-lg p-6 shadow-md border border-blue-100">
            <h3 className="text-xl font-semibold text-blue-700 mb-4">
              🎯 Prediction
            </h3>

            {prediction && confidence !== null ? (
              <>
                {/* Main Prediction */}
                <div className="mb-4">
                  <p className="text-lg font-medium text-gray-800">
                    <span className="font-semibold text-blue-700">Predicted Class:</span> {prediction}
                  </p>
                  <p className="text-sm text-gray-600">
                    <span className="font-semibold">Confidence:</span> {(confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </>
            ) : (
              <p className="text-gray-500">No prediction made yet.</p>
            )}

            {/* Top 3 Predictions */}
            {topPredictions && topPredictions.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm text-gray-500 font-semibold mb-2">Top 3 Predictions:</h4>
                <ul className="space-y-1">
                  {topPredictions.map((item, index) => (
                    <li key={index} className="flex justify-between text-sm text-gray-700">
                      <span>{item.label}</span>
                      <span>{(item.confidence * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Bottom Section: Upload Again Button */}
        <div className="mt-10 text-center">
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition"
            onClick={handleButtonClick}
          >
            Upload New Audio
          </button>
        </div>
      </div>
    </section>
  );
};

export default NoisePredictionResult;
