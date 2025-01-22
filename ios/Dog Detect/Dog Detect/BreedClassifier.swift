import CoreML
import Vision

class BreedClassifier {
    static func classifyImage(_ image: UIImage, completion: @escaping (String?) -> Void) {
        guard let model = try? VNCoreMLModel(for: DogBreedClassifier().model) else {
            completion(nil)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
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
        try? handler.perform([request])
    }
}