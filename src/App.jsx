import React, { useRef, useState, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { drawKeypoints, drawSkeleton } from "./utils";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);

  useEffect(() => {
    const runMoveNet = async () => {
      tf.ready();
      const net = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        }
      );
      setDetector(net);
    };
    runMoveNet();

    return () => {
      if (detector) {
        detector.dispose();
      }
    };
  }, []);
  const debounce = (fn, delay) => {
    let timeoutID;
    return function (...args) {
      if (timeoutID) {
        clearTimeout(timeoutID);
      }
      timeoutID = setTimeout(() => {
        fn(...args);
      }, delay)
    }
  }
  const captureImage =(() => {
    if (webcamRef.current) {
      setCapturedImage(webcamRef.current.getScreenshot());
    }
  });

  const stopModel = () => {
    if (detector) {
      cancelAnimationFrame(detect);
      detector.dispose();
      setDetector(null);
    }
  };

  function calculateAngle([xA, yA], [xB, yB], [xC, yC]) {
    // Usando la fórmula de la distancia entre dos puntos para calcular la longitud de los lados del triángulo
    const AB = Math.sqrt(Math.pow(xB - xA, 2) + Math.pow(yB - yA, 2));
    const BC = Math.sqrt(Math.pow(xB - xC, 2) + Math.pow(yB - yC, 2));
    const AC = Math.sqrt(Math.pow(xC - xA, 2) + Math.pow(yC - yA, 2));
    // Usando la ley de cosenos para calcular el ángulo entre los lados AB y AC
    return Math.acos((BC * BC + AB * AB - AC * AC) / (2 * BC * AB));
  }

  const detect = async () => {
    if (!detector) return;
    if (
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4 &&
      detector !== null
    ) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      video.width = videoWidth;
      video.height = videoHeight;

      try {
        const poses = await detector.estimatePoses(video);
        drawCanvas(poses, video, videoWidth, videoHeight, canvasRef);
      } catch (error) {
        console.error("Pose estimation error: ");
      }
    }
    requestAnimationFrame(detect);
  };

  const drawCanvas = (poses, video, videoWidth, videoHeight, canvasRef) => {
    const canvas = canvasRef.current;
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext("2d");
    poses.forEach(({ keypoints }) => {
      drawKeypoints(keypoints, 0.6, ctx);
      drawSkeleton(keypoints, 0.6, ctx);
      if (captureImage) {
      

        for (const dir of ["left", "right"]) {
          const shoulder = keypoints.find((k) => k.name === `${dir}_shoulder`);
          const elbow = keypoints.find((k) => k.name === `${dir}_elbow`);
          const wrist = keypoints.find((k) => k.name === `${dir}_wrist`);
          if (shoulder.score > 0.6 && elbow.score > 0.6 && wrist.score > 0.6) {
            // Calcula el ángulo del brazo
            const angle = calculateAngle(
              [shoulder.x, shoulder.y],
              [elbow.x, elbow.y],
              [wrist.x, wrist.y]
            );
            // Si el ángulo cumple con ciertos criterios, captura una imagen del video π/4 radianes es igual a 45 grados y π/2 radianes es igual a 90 grados.

            if (angle > Math.PI / 4 && angle < Math.PI / 2) {
              // const image = canvas.toDataURL('image/png');
              captureImage();
            }
          }
        }
      }
    });
  };

  useEffect(() => {
    if (detector !== null) {
      detect();
    }
  }, [detector]);

  return (
    <div className="flex flex-col items-center min-h-screen">
      <header className="w-full">
        <nav className="bg-slate-900 text-white p-4 flex justify-between items-center">
          <h1 className="text-2xl">Pose Detection</h1>
        </nav>
      </header>
      <div className="container mx-auto text-center my-4">
        <h1 className="text-4xl">
          {detector ? "Pose Detection TF JS" : "Cargando modelo"}
        </h1>
      </div>
      <div className="flex flex-row justify-center items-center space-x-4">
        <div className="relative">
          <Webcam ref={webcamRef} className="w-full h-auto" />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
          />
        </div>
        {capturedImage && (
          <div className="flex justify-center mt-4">
            <img
              src={capturedImage}
              alt="captured"
              className="border border-gray-300"
            />
          </div>
        )}
      </div>
      <div className="flex justify-center space-x-4 my-4">
        <button
          onClick={captureImage}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Capturar
        </button>
        <button
          onClick={() => setCapturedImage(null)}
          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
        >
          Limpiar
        </button>
      </div>
    </div>
  );
}

export default App;
