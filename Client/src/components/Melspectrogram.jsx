import React from "react";
export default function Melspectrogram(){
    return (
        <div class=" bg-gradient-to-b from-blue-50 to-white py-2 px-4 flex flex-col items-center justify-center text-center">
          
        <h1 class="text-3xl sm:text-4xl font-bold text-blue-800 mb-4 animate-fade-in mt-20">
          Audio to Mel-Spectrogram
        </h1>
        <p class="text-gray-600 max-w-xl mb-10 animate-fade-in delay-200">
          See how your audio transforms into a colorful frequency map. A visual journey from sound to signal.
        </p>

      
        <div class="flex space-x-1 items-end mb-10 animate-wave">
          <div class="w-2 h-8 bg-blue-500 rounded"></div>
          <div class="w-2 h-12 bg-blue-400 rounded"></div>
          <div class="w-2 h-6 bg-blue-600 rounded"></div>
          <div class="w-2 h-10 bg-blue-300 rounded"></div>
          <div class="w-2 h-7 bg-blue-500 rounded"></div>
          <div class="w-2 h-14 bg-blue-400 rounded"></div>
          <div class="w-2 h-8 bg-blue-600 rounded"></div>
        </div>


        <div class="flex items-center justify-center mb-10 space-x-3">
          <span class="w-3 h-3 bg-blue-600 rounded-full animate-bounce"></span>
          <span class="w-3 h-3 bg-blue-600 rounded-full animate-bounce delay-150"></span>
          <span class="w-3 h-3 bg-blue-600 rounded-full animate-bounce delay-300"></span>
          <span class="text-blue-600 font-medium ml-2">Converting...</span>
        </div>


        <div class="w-full max-w-3xl h-64 bg-gradient-to-r from-purple-200 via-pink-200 to-yellow-100 rounded-xl shadow-md flex items-center justify-center border border-dashed border-purple-400">
          <span class="text-purple-700 font-bold text-lg tracking-wide">Mel-Spectrogram Output</span>
        </div>
      </div>
    )
}