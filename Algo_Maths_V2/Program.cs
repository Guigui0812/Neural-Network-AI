using System;
using System.Collections.Generic;
using System.IO;

namespace Algo_Maths_V2
{
    class Program
    {
        // Procédure permettant de calculer la sigmoid de la valeur passée en paramètre
        static double Sigmoid(double nb)
        {
            return 1.0 / (1.0 + Math.Exp(-nb));
        }
        // Procédure permettant de calculer la dérivée de la sigmoid
        static double SigmoidPrime(double nb)
        {
            return nb * (1.0 - nb);
        }
        // Phase de propagation
        static double[] Propagation(List<double> W, List<double> entreeData, double[] neuronesValues)
        {
            // Calcule des éléments du réseau : x[A], x[B] et s[Z]
            neuronesValues[0] = Sigmoid(1 * W[0] + entreeData[0] * W[1] + entreeData[1] * W[2]); // x[A]
            neuronesValues[1] = Sigmoid(1 * W[3] + entreeData[0] * W[4] + entreeData[1] * W[5]); // x[B]
            neuronesValues[2] = Sigmoid(1 * W[6] + neuronesValues[0] * W[7] + neuronesValues[1] * W[8]); // s[Z]
            return neuronesValues;
        }
        // Phase de rétro-propagation
        static List<double> RetroPropagation(List<double> W, List<double> R, List<double> entreeData, double[] neuronesValues, int k, double pas)
        {
            // Calcul du delta du neurone Z en appliquant la formule : delta[Z] <- (R[k] -s[Z]) *f0(x[Z])
            double DeltaZ = (R[k] - neuronesValues[2]) * SigmoidPrime(neuronesValues[2]);
            // Calcul des deltas des neurones A et B en appliquant la formule : sum( W[N,N']*delta[N'] ) *f0(x[N])
            double DeltaB = (DeltaZ * W[8]) * SigmoidPrime(neuronesValues[1]);
            double DeltaA = (DeltaZ * W[7]) * SigmoidPrime(neuronesValues[0]);
            // Mise à jour des poids du réseau en appliquant la formule : W[N,N'] <- W[N,N '] + pas * delta[N'] *s[N]
            
            // Poids vers Z
            W[8] += DeltaZ * neuronesValues[1] * pas;
            W[7] += DeltaZ * neuronesValues[0] * pas;
            W[6] += DeltaZ * 1 * pas;
            // Poids vers B
            W[5] += DeltaB * entreeData[1] * pas;
            W[4] += DeltaB * entreeData[0] * pas;
            W[3] += DeltaB * 1 * pas;

            // Poids vers A
            W[2] += DeltaA * entreeData[1] * pas;
            W[1] += DeltaA * entreeData[0] * pas;
            W[0] += DeltaA * 1 * pas;
            return W;
        }
        static void EntrainerReseau(List<List<double>> E, List<double> R, List<double> W,
       int nbNeurones, int Tmax, double pas, double[] neuronesValues, int nbData)
        {
            Random rand = new Random();
            // Entrainement du réseau jusqu'à Tmax
            for (int i = 0; i < Tmax; i++)
            {
                // Sélection d'un cas :
                int k = rand.Next(nbData);
                List<double> entreeData = new List<double>(E[k]);
                // Phase de propagation :
                neuronesValues = Propagation(W, entreeData, neuronesValues);
                // Phase de retro-propagation :
                W = RetroPropagation(W, R, entreeData, neuronesValues, k, pas);
            }
        }
        // Procédure permettant de tester le réseau après entrainement.
        static void TesterReseau(List<List<double>> E, List<double> W, List<double> R, double[] neuronesValues)
        {
            // Calcul de la sortie pour toutes les valeurs du csv.
            for (int i = 0; i < E.Count; i++)
            {
                List<double> entreeData = new List<double>(E[i]);
                // Activation des neurones avec la Sigmoid.
                neuronesValues[0] = Sigmoid(1 * W[0] + entreeData[0] * W[1] + entreeData[1
               ] * W[2]);
                neuronesValues[1] = Sigmoid(1 * W[3] + entreeData[0] * W[4] + entreeData[1
               ] * W[5]);
                neuronesValues[2] = Sigmoid(1 * W[6] + neuronesValues[0] * W[7] + neuronesValues[1] * W[8]);
                // Affichage de la sortie de Z
                Console.WriteLine(" - Sortie Z pour e1 = " + entreeData[0] + " et e2 = " +
               entreeData[1] + " : obtenu --> " + neuronesValues[2] + " / attendu --> " + R[i]);
            }
        }
        // Procédure générale comportant les différentes étapes de notre réseau.
        static void ReseauNeurones(List<List<double>> E, List<double> R, List<double> W, int nbNeurones, int Tmax, double pas, int nbData)
        {
            // Tableau représentant la valeur des différents neurones : A, B et Z ici caron a 3 neurones-- > tableau de taille 3.
            double[] neuronesValues = new double[nbNeurones];
            Console.WriteLine("Poids initiaux : ");
            int i = 1; // Permet d'afficher le nom du poids
                       // Affichage des poids initiaux.
            foreach (double weight in W)
            {
                Console.WriteLine(" - W" + i + " : " + weight);
                i++;
            }
            Console.WriteLine("\nPoids finaux: ");
            // Fonction permettant d'entrainer le réseau
            EntrainerReseau(E, R, W, nbNeurones, Tmax, pas, neuronesValues, nbData);
            i = 1;
            // Affichage des poids ajustés après entrainement
            foreach (double weight in W)
            {
                Console.WriteLine(" - W" + i + " : " + weight);
                i++;
            }
            Console.WriteLine("\nSorties du neurone Z :");
            // Redéfinition du tableau contenant les valeurs des neurones.
            neuronesValues = new double[nbNeurones];
            // Appel de la fonction de test du réseau.
            TesterReseau(E, W, R, neuronesValues);
        }
        // Lecture du fichier de données et création des listes d'entrées et de réponses.
        static List<List<double>> CsvReader(List<List<double>> E, List<double> R)
        {
            using (StreamReader reader = new StreamReader("../../../donnees.csv"))
            {
                // Listes temporaires permettant de stocker les valeurs.
                List<string> e1Tmp = new List<string>();
                List<string> e2Tmp = new List<string>();
                List<string> RTmp = new List<string>();
                int cpt = 0;
                string[] values = new string[3];
                // Lecture du csv
                while (!reader.EndOfStream)
                {
                    List<double> dataTmp = new List<double>();
                    string line = reader.ReadLine();
                    values = line.Split(';');
                    if (cpt != 0) // Permet d'ignorer la première ligne
                    {
                        // Placement des entrées sous forme de couples dans E, et des réponses dans R.
                        dataTmp.Add(Convert.ToDouble(values[0]));
                        dataTmp.Add(Convert.ToDouble(values[1]));
                        R.Add(Convert.ToDouble(values[2]));
                        E.Add(new List<double>(dataTmp));
                    }
                    cpt++;
                }
            }

            return E;
        }

        // Générateur de poids aléatoires pour l'ensemble des liasons du réseau.
        static List<double> GenerateWeights(List<double> W, int nbNeurones)
        {
            Random rand = new Random();
            int tmpSign;
            double tmpRand;
            for (int i = 0; i < nbNeurones * 3; i++)
            {
                tmpRand = rand.NextDouble();

                // Code permettant d'avoir des poids négatifs.
                tmpSign = rand.Next(2);

                if (tmpSign == 1)
                {
                    tmpRand = tmpRand * -1;
                }
                W.Add(tmpRand);
            }
            return W;
        }
        
        static void Main(string[] args)
        {
            List<double> R = new List<double>(); // Liste des réponses attendues
            List<double> W = new List<double>(); // Liste des poids du réseau
            
            // Liste des valeurs d'entrées sous forme de liste de liste pour sélectionner les couples de valeurs e1 et e2.
            List<List<double>> E = new List<List<double>>();
            const int nbNeurones = 3;
            const int nbData = 30;
            int Tmax = 50000;
            double pas = 0.05;
            // Récupération des données du csv
            E = CsvReader(E, R);
            // Génération de poids aléatoires.
            W = GenerateWeights(W, nbNeurones);
            // Contient les différentes opérations réalisées pour "résoudre" le réseau multicouche
            ReseauNeurones(E, R, W, nbNeurones, Tmax, pas, nbData);
        }
    }
}