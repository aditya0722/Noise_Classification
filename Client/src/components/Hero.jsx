import React, { useRef, useState } from "react";
import sound from "../images/sound-wave.svg";

export default function Hero({fileInputRef,handleButtonClick,handleFileChange,fileName}) {
 
    

  

  return (
    <>
      <section className="flex flex-col md:flex-row items-center justify-evenly p-10 bg-blue-50">
        <div className="max-w-xxl">
          <h1 className="text-4xl font-bold text-blue-700">
            Classify Environmental Noise in Real Time
          </h1>
          <p className="mt-4 text-gray-600">
            Upload a sample to identify noise types like
            Airconditionar, Jackhammer, Dog Bark, and more.
          </p>
          <div className="mt-6 space-x-4">
            <button
              onClick={handleButtonClick}
              className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition"
            >
              Upload Audio
            </button>
            <input
              type="file"
              accept=".mp3, .wav, .WAV"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
          {fileName && (
            <p className="mt-3 text-sm text-green-600">Selected file: {fileName}</p>
          )}
        </div>
        <img src={sound} className="w-64 mt-8 md:mt-0" alt="Sound Wave" />
      </section>
     
    </>
  );
}
