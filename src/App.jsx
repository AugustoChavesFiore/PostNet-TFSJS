import React, { useRef, useState, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import { drawKeypoints, drawSkeleton } from "./utilities";
import Webcam from "react-webcam";



function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [fullBodyVisible, setFullBodyVisible] = useState(0);
  const [capturedImage, setCapturedImage] = useState(null);

  useEffect(() => {
    const runMoveNet = async () => {
      const net = await posenet.load({
        inputResolution: { width: 640, height: 480 },
        scale: 0.8,
      });
      setDetector(net);
    };
    runMoveNet();
  
    return () => {
      if (detector) {
        detector.dispose();
      }
    };
  }, []);

  const captureImage = () => {
    if (webcamRef.current) {
      setCapturedImage(webcamRef.current.getScreenshot());
    }
  };

  const stopModel = () => {
    if (detector) {
      cancelAnimationFrame(detect)
      detector.dispose(); 
      setDetector(null);
    }
  };

  


  const checkFullBodyVisible = (keypoints) => {
    const fullBody = [
      "nose",
      "leftEye",
      "rightEye",
      "leftEar",
      "rightEar",
      "leftShoulder",
      "rightShoulder",
      "leftElbow",
      "rightElbow",
      "leftWrist",
      "rightWrist",
      "leftHip",
      "rightHip",
      "leftKnee",
      "rightKnee",
      "leftAnkle",
      "rightAnkle",
    ];

     const verify = fullBody.every(part => {
      const point = keypoints.find(point => point.part === part);
      return point && point.score >= 0.6 && point.position.x !== 0 && point.position.y !== 0;
    });
    if(verify){
      captureImage();
      stopModel();
    }

  };




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
        const pose = await detector.estimateSinglePose(video);
        if (pose) {
          drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
          if (checkFullBodyVisible(pose.keypoints)) {
            setFullBodyVisible(prevCount => prevCount + 1);
            console.log('Full body visible');
          }
        } else {
          console.log("No poses detected");
        }
      } catch (error) {
        console.error("Pose estimation error: ");
      }
    }
    requestAnimationFrame(detect);
  };


  const drawCanvas = (pose, video, videoWidth, videoHeight, canvasRef) => {
    const canvas = canvasRef.current;
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext("2d");

    drawKeypoints(pose.keypoints, 0.8, ctx);
    drawSkeleton(pose.keypoints, 0.8, ctx);
  };

  useEffect(() => {
    if (detector !== null) {
      detect();
    }
  }, [detector]);

  return (
    <div>
    <header>
      <nav className="bg-slate-900 text-white p-4 flex justify-between items-center">
        <h1 className="text-2xl">Pose Detection</h1>
      </nav>
    </header>
    <div className="container mx-auto">
      <h1 className="text-4xl text-center my-4">Pose Detection posenet model TF JS</h1>
      {/* <p className="text-center my-4">
        Full body visible: {fullBodyVisible}
      </p> */}
    </div>
    {
      capturedImage ? (
        <div className="container mx-auto">
          <h2>Se detecto una persona </h2>
          <img src={capturedImage} alt="Captured" />
        </div>
      ):(
        <div className="relative p-4">
      <Webcam
        className="absolute"
        ref={webcamRef}
      />
      <canvas
        className="absolute"
        ref={canvasRef}
      />
    </div>
      )
    }
  </div>
  );
}


export default App;
