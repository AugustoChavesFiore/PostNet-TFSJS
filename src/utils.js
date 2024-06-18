import * as poseDetection from '@tensorflow-models/pose-detection'; 

export function drawKeypoints(keypoints, minConfidence, ctx) {
    
    keypoints.forEach((keypoint) => {
      if (keypoint.score > minConfidence) {
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
      }
    });
  }

  export function drawSkeleton(keypoints, minConfidence, ctx,) {

    const adjacentKeyPoints = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);
  
    adjacentKeyPoints.forEach((pair) => {
      const [i, j] = pair;
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];
  
    
      if (kp1.score > minConfidence && kp2.score > minConfidence) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  }