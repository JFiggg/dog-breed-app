//
//  BreedClassifier.swift
//  Dog Detect
//
//  Created by Jason Figueroa on 1/21/25.
//


import CoreML
import Vision
import UIKit
import CoreImage

struct BreedClassifier {
    static func classifyImage(_ image: UIImage, completion: @escaping (String?) -> Void) {
        do {
            // Use configuration as recommended
            let config = MLModelConfiguration()
            let model = try DogBreedClassifier(configuration: config)
            let visionModel = try VNCoreMLModel(for: model.model)
            
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                guard let results = request.results as? [VNClassificationObservation],
                      let topResult = results.first else {
                    completion(nil)
                    return
                }
                completion(topResult.identifier)
            }
            
            guard let ciImage = CIImage(image: image) else {
                completion(nil)
                return
            }
            
            let handler = VNImageRequestHandler(ciImage: ciImage)
            try handler.perform([request])
            
        } catch {
            print("Classification error: \(error)")
            completion(nil)
        }
    }
}
