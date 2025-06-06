import './App.css'
import Navbar from './components/navbar'
import Hero from './components/Hero'
import Melspectrogram from './components/Melspectrogram'
import NoisePredictionResult from './components/NoisePredictionResult'
import Footer from './components/Footer'
import { ScaleLoader } from "react-spinners";
import { useState,useRef } from "react";

function App() {
  const fileInputRef = useRef(null);
  const [isloading,setIsloding]=useState(false);
  const [PredictedData,setPredictedData]=useState(null);
  const [fileName, setFileName] = useState("");
  const [errorMessage, setErrorMessage] = useState(""); // State for user-facing error messages

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };
  
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    setErrorMessage(""); // Clear previous error messages

    if (file && (file.type === "audio/mp3" || file.type === "audio/wav" || file.type === "audio/mpeg" || file.type === "audio/flac" || file.type === "audio/ogg")) { // Added more audio types
      setFileName(file.name);
      setIsloding(true);
      setPredictedData(null); // Clear previous prediction data

      const formData = new FormData();
      // --- IMPORTANT CHANGE 1: Match backend's expected parameter name 'audio_file' ---
      formData.append("audio_file", file); 
      
      try {
        // --- IMPORTANT CHANGE 2: Match backend's endpoint '/upload-audio' ---
        const response = await fetch("https://8000-cs-a315db39-7433-4f72-a3a8-30d2de187983.cs-asia-east1-cats.cloudshell.dev/upload-audio", {
          method: "POST",
          // No 'Content-Type' header needed for FormData; browser sets it automatically
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          console.error("Server error response:", errorData);
          setErrorMessage(errorData.detail || "An unknown error occurred on the server.");
          return; // Stop execution if response is not OK
        }
        
        const data = await response.json();
        setPredictedData(data)
        console.log("Prediction from server:", data);
      } catch (error) {
        console.error("Error uploading file:", error);
        setErrorMessage("Failed to connect to the server or process the file. Please try again.");
      } finally {
        setIsloding(false);
      }
    } else {
      // --- IMPORTANT CHANGE 3: Replaced alert() with state-based message ---
      setErrorMessage("Please upload a valid audio file (.mp3, .wav, .mpeg, .flac, .ogg).");
      setFileName(""); // Clear file name if invalid
    }
  };
  
  return (
    <>
      <Navbar/>

      <Hero 
        setIsloding={setIsloding} 
        fileInputRef={fileInputRef} 
        handleButtonClick={handleButtonClick} 
        handleFileChange={handleFileChange} 
        fileName={fileName}
        errorMessage={errorMessage} // Pass error message to Hero component if it displays it
      />
      
      {isloading && (
        <div className="fixed inset-0 bg-blue-10 backdrop-blur-sm z-50 flex items-center justify-center flex-col">
          <ScaleLoader color="black" />
          <h1 className='text-blue-800'>Loading...</h1> 
        </div>
      )}
      
      {/* Display error message if present */}
      {errorMessage && (
        <div className="fixed inset-0 bg-red-100 bg-opacity-75 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-xl text-red-700 text-center">
            <h2 className="text-xl font-bold mb-4">Error!</h2>
            <p className="mb-4">{errorMessage}</p>
            <button 
              onClick={() => setErrorMessage("")} 
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {PredictedData && (
        <NoisePredictionResult
          // Assuming PredictedData now directly contains predicted_noise_type and confidence
          spectrogramUrl={PredictedData.spectrogram_url}
          prediction={PredictedData.predicted_noise_type}
          confidence={PredictedData.confidence}
          topPredictions={PredictedData.top_3_predictions} // Assuming this is an array of objects with label and confidence
          handleButtonClick={handleButtonClick}
        />
      )}
      
      <Melspectrogram/> 
      <Footer/>
    </>
  )
}

export default App;
