//
//  DogBreedInfo.swift
//  Dog Detect
//
//  Created by Jason Figueroa on 1/21/25.
//


import SwiftUI

struct ResultsView: View {
    let image: UIImage
    let breedName: String  // This will get the predicted breed name
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Breed Name
                Text(breedName)
                    .font(.system(size: 32, weight: .bold))
                    .foregroundColor(.blue)
                
                // Image
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
                    .cornerRadius(15)
                    .shadow(radius: 5)
                
                Spacer()
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    // This is for the Xcode preview
    ResultsView(
        image: UIImage(systemName: "dog")!,
        breedName: "German Shepherd"
    )
}
