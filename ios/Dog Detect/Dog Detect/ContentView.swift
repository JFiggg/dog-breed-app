//
//  ContentView.swift
//  Dog Detect
//
//  Created by Jason Figueroa on 1/21/25.
//

import SwiftUI
import UIKit
import Vision
import CoreML

struct ContentView: View {
    @State private var isShowingImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var sourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var showingResults = false
    @State private var predictedBreed: String = ""
    
    private var model: DogBreedClassifier? = {
        do {
            let config = MLModelConfiguration()
            let model = try DogBreedClassifier(configuration: config)
            print("Model loaded successfully")
            return model
        } catch {
            print("Error loading model: \(error)")
            return nil
        }
    }()
    
    func classifyImage(_ image: UIImage) {
        print("Starting classification...")
        
        guard let preprocessedBuffer = image.preprocessForInference() else {
            print("Failed to preprocess image")
            return
        }
        
        do {
            let config = MLModelConfiguration()
            let model = try DogBreedClassifier(configuration: config)
            
            let input = DogBreedClassifierInput(input_1: preprocessedBuffer)
            let output = try model.prediction(input: input)
            
            let outputArray = output.var_1146
            var predictions: [(index: Int, probability: Float)] = []
            
            // Apply softmax to convert logits to probabilities
            let values = (0..<outputArray.count).map { Float(outputArray[$0].floatValue) }
            let maxValue = values.max()!
            let expValues = values.map { exp($0 - maxValue) }  // Subtract max for numerical stability
            let sumExp = expValues.reduce(0, +)
            let softmax = expValues.map { $0 / sumExp }
            
            // Create predictions array
            for (index, probability) in softmax.enumerated() {
                predictions.append((index, probability))
            }
            
            // Sort by probability
            predictions.sort { $0.probability > $1.probability }
            
            // Print top 5 predictions
            print("\nTop 5 predictions:")
            for i in 0..<min(5, predictions.count) {
                let pred = predictions[i]
                print("\(BreedNames.getBreedName(for: pred.index)): \(pred.probability * 100)%")
            }
            
            // Use the highest probability prediction
            let topPrediction = predictions[0]
            let breedName = BreedNames.getBreedName(for: topPrediction.index)
            
            DispatchQueue.main.async {
                self.predictedBreed = breedName
                self.showingResults = true
            }
        } catch {
            print("Classification error:", error)
        }
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Title Section
            VStack(spacing: 8) {
                Text("Dog Detect")
                    .font(.system(size: 40, weight: .bold))
                    .foregroundColor(.blue)
                
                Text("Discover your dog's breed instantly!")
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }
            .padding(.top, 30)
            
            // Image Display Section
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .cornerRadius(15)
                    .shadow(radius: 5)
                
                Button(action: {
                    classifyImage(image)
                    print("Analyze button pressed")
                }) {
                    Text("Analyze Breed")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
            } else {
                // Placeholder when no image is selected
                VStack {
                    Image(systemName: "dog")
                        .font(.system(size: 80))
                        .foregroundColor(.blue.opacity(0.5))
                    Text("No image selected")
                        .foregroundColor(.gray)
                }
                .frame(height: 300)
            }
            
            // Buttons Section
            HStack(spacing: 30) {
                Button(action: {
                    self.sourceType = .camera
                    self.isShowingImagePicker = true
                }) {
                    VStack {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 30))
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Color.blue)
                            .clipShape(Circle())
                            .shadow(radius: 5)
                        
                        Text("Take Photo")
                            .font(.headline)
                            .foregroundColor(.blue)
                    }
                }
                
                Button(action: {
                    self.sourceType = .photoLibrary
                    self.isShowingImagePicker = true
                }) {
                    VStack {
                        Image(systemName: "photo.fill")
                            .font(.system(size: 30))
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Color.blue)
                            .clipShape(Circle())
                            .shadow(radius: 5)
                        
                        Text("Upload Photo")
                            .font(.headline)
                            .foregroundColor(.blue)
                    }
                }
            }
            .padding(.top, 20)
            
            // Footer Text
            Text("We can identify over 120 dog breeds!")
                .font(.caption)
                .foregroundColor(.gray)
                .padding(.top, 30)
            
            Spacer()
        }
        .padding()
        .sheet(isPresented: $isShowingImagePicker) {
            ImagePicker(selectedImage: $selectedImage, sourceType: sourceType)
        }
        .sheet(isPresented: $showingResults) {
            if let image = selectedImage {
                ResultsView(image: image, breedName: predictedBreed)
            }
        }
    }
}

#Preview {
    ContentView()
}
