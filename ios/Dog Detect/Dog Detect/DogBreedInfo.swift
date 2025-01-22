//
//  DogBreedInfo.swift
//  Dog Detect
//
//  Created by Jason Figueroa on 1/21/25.
//


import SwiftUI

struct DogBreedInfo {
    let breed: String
    let stats: [(String, String)] // [(Stat name, value)]
    let description: String
}

struct ResultsView: View {
    let image: UIImage
    @State private var breedInfo = DogBreedInfo(
        breed: "German Shepherd",
        stats: [
            ("Intelligence", "5/5"),
            ("Energy Level", "4/5"),
            ("Life Expectancy", "9-13 years"),
            ("Trainability", "5/5"),
            ("Shedding", "High")
        ],
        description: "German Shepherds are intelligent, capable, and loving dogs. Originally developed to herd flocks, they are now one of the most popular breeds for both work and family companionship. They excel in various roles including police, military, and service work."
    )
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Breed Name
                Text(breedInfo.breed)
                    .font(.system(size: 32, weight: .bold))
                    .foregroundColor(.blue)
                
                // Image
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
                    .cornerRadius(15)
                    .shadow(radius: 5)
                
                // Stats Section
                VStack(spacing: 15) {
                    Text("Breed Characteristics")
                        .font(.headline)
                        .padding(.bottom, 5)
                    
                    ForEach(breedInfo.stats, id: \.0) { stat in
                        HStack {
                            Text(stat.0)
                                .foregroundColor(.gray)
                            Spacer()
                            Text(stat.1)
                                .bold()
                        }
                        .padding(.horizontal)
                        Divider()
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(15)
                
                // Description
                VStack(alignment: .leading) {
                    Text("About")
                        .font(.headline)
                        .padding(.bottom, 5)
                    
                    Text(breedInfo.description)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding()
                
                Spacer()
            }
            .padding()
        }
    }
}

#Preview {
    ResultsView(image: UIImage(systemName: "dog")!)
}