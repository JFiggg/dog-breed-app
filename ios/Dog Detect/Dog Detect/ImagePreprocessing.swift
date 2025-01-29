//
//  ImagePreprocessing.swift
//  Dog Detect
//
//  Created by Jason Figueroa on 1/24/25.
//

import UIKit
import CoreML

extension UIImage {
    func preprocessForInference() -> CVPixelBuffer? {
        // Step 1: Resize to 224x224
        let imageSize = CGSize(width: 224, height: 224)
        UIGraphicsBeginImageContextWithOptions(imageSize, true, 1.0)
        self.draw(in: CGRect(origin: .zero, size: imageSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        // Step 2: Convert to CVPixelBuffer
        guard let pixelBuffer = try? resizedImage.toCVPixelBuffer() else {
            return nil
        }
        
        return pixelBuffer
    }
}
