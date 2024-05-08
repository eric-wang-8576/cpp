#pragma once
#include <iostream>
#include "randomgenerator.hpp"

struct Config {
  int address;
  bool personaNonGrata;
  bool acceptingRange;
  int addressBegin;
  int addressEnd;
  Config() {}
  Config(int address, bool personaNonGrata, bool acceptingRange,
    int addressBegin, int addressEnd) {
    this->address = address;
    this->personaNonGrata = personaNonGrata;
    this->acceptingRange = acceptingRange;
    this->addressBegin = addressBegin;
    this->addressEnd = addressEnd;
  }
};

struct Header {
  int source;
  int dest;
  int sequenceNumber;
  int trainSize;
  int tag;
  Header() {}
  Header(int source, int dest, int seq, int trainSize, int tag) {
    this->source = source;
    this->dest = dest;
    this->sequenceNumber = seq;
    this->trainSize = trainSize;
    this->tag = tag;
  }
};

struct Body {
  long iterations;
  long seed;
  Body() {
    iterations = 0;
    seed = 0;
  }
  Body(long iterations, long seed) {
    this->iterations = iterations;
    this->seed = seed;
  }
};

struct AddressPair {
  int source;
  int dest;
  AddressPair () {}
  AddressPair(int source, int dest) { 
    this->source = source;
    this->dest = dest;
  }
};

enum class MessageType {
    ConfigPacket, 
    DataPacket
};

struct Packet {
  Config config;
  Header header;
  Body body;
  MessageType type;
  Packet(Config config) : config(config) {
    this->type = MessageType::ConfigPacket;
  } 
  Packet(Header header, Body body) : header(header), body(body) {
    this->type = MessageType::DataPacket;
  }
  void printPacket() {
    if( type == MessageType::ConfigPacket ) {
        std::cout << "CONFIG: " << config.address << " <" << config.personaNonGrata <<
        "," << config.acceptingRange << ">" << " [" << config.addressBegin <<
        "," << config.addressEnd << ")" << std::endl;
    }
    else {
        std::cout << "data:   " << "<" << header.source << "," << header.dest <<
        ">" << " " << header.sequenceNumber << "/" << header.trainSize << " (" <<
        header.tag << ")" << std::endl;
    }
  }
};

struct PacketStruct {
  AddressPair pair;
  int trainSize;
  int totalTrains;
  double meanWork;
  int tag;
  int sequenceNumber = 0;
  int trainNumber = 0;

  PacketStruct() {}
  PacketStruct(AddressPair pair, int trainSize, int totalTrains,
    double meanWork, int tag) : pair(pair) {
    this->trainSize = trainSize;
    this->totalTrains = totalTrains;
    this->meanWork = meanWork;
    this->tag = tag;
  }
};

class AddressPairGenerator {
  double speed;
  int mask;
  int logSize;
  int source;
  int dest;
  double sourceResidue;
  double destResidue;
  ExponentialGenerator expGen;
  UniformGenerator uniGen;

public:
  AddressPairGenerator(int meanCommsPerAddress, int logSize, double mean) 
    : expGen(mean)
  {
    this->speed = 2.0 / ((double) meanCommsPerAddress);
    this->mask = (1 << logSize) - 1;
    this->logSize = logSize;
    this->source = 0;
    this->dest = 0;
    this->sourceResidue = 0.0;
    this->destResidue = 0.0;
  }

  AddressPair getPair() {
    sourceResidue = sourceResidue + speed*uniGen.getUnitRand();
    destResidue = destResidue + speed*uniGen.getUnitRand();
    while( sourceResidue > 0.0 ) {
      source = ( source + 1 ) & mask;
      sourceResidue = sourceResidue - 1.0;
    }
    while( destResidue > 0.0 ) {
      dest = ( dest + mask ) & mask; // he's walking backward...
      destResidue = destResidue - 1.0;
    }
    return AddressPair(uniGen.mangle(1+((source+expGen.getRand())))& mask,
                          uniGen.mangle(1+((dest+expGen.getRand())))& mask);
  }
};

class PacketGenerator {
  AddressPairGenerator pairGen;
  ExponentialGenerator expGen;
  UniformGenerator uniGen;
  int mask; // numTrains - 1
  int addressesMask;
  double meanTrainSize;
  double meanTrainsPerComm;
  double meanWork;
  double pngFraction;
  double acceptingFraction;
  int timeToNextConfigPacket = 0;
  int lastConfigAddress;
  int numConfigPackets = 0;
  int configAddressMask;
  std::vector<PacketStruct> trains;
public:
  PacketGenerator(
    int numAddressesLog,
    int numTrainsLog,
    double meanTrainSize,
    double meanTrainsPerComm,
    int meanWindow,
    int meanCommsPerAddress,
    int meanWork,
    double configFraction,
    double pngFraction,
    double acceptingFraction ) 
    : expGen((1.0/configFraction) - 1),
      pairGen(meanCommsPerAddress, numAddressesLog, (double) meanWindow)
  {
    this->mask = (1 << numTrainsLog) - 1;
    this->addressesMask = (1 << numAddressesLog) - 1;
    this->meanTrainSize = meanTrainSize;
    this->meanTrainsPerComm = meanTrainsPerComm;
    this->meanWork = (double) meanWork;
    this->lastConfigAddress = pairGen.getPair().source;
    this->configAddressMask = (1 << (numAddressesLog >> 1)) - 1;
    this->pngFraction = pngFraction;
    this->acceptingFraction = acceptingFraction;
    this->trains.resize(mask + 1);
    for( int i = 0; i <= mask; i++ ) {
      PacketStruct ps {pairGen.getPair(), 
        expGen.getRand(meanTrainSize),expGen.getRand(meanTrainsPerComm), 
        static_cast<double>(expGen.getRand(this->meanWork)), uniGen.getRand()};
      this->trains[i] = ps;
    }
  }

  Packet getPacket() {
    if( timeToNextConfigPacket == 0 ) {
      numConfigPackets++;
      timeToNextConfigPacket = expGen.getRand();
      return getConfigPacket();
    }
    else
      return getDataPacket();
  }

  Packet getConfigPacket() {
    lastConfigAddress = pairGen.getPair().source;
    int addressBegin = uniGen.getRand(addressesMask-configAddressMask);
    return Packet(Config(lastConfigAddress, uniGen.getUnitRand() < pngFraction, 
      uniGen.getUnitRand() < acceptingFraction, addressBegin, 
      uniGen.getRand(addressBegin+1,addressBegin+configAddressMask)));  
  }

  Packet getDataPacket() {
    if( timeToNextConfigPacket > 0 ) 
      timeToNextConfigPacket--;
    int trainIndex = uniGen.getRand() & mask;
    PacketStruct pkt = trains[trainIndex];
    Packet packet = Packet(
      Header(pkt.pair.source, pkt.pair.dest, pkt.sequenceNumber, pkt.trainSize, pkt.tag),
      Body(expGen.getRand(pkt.meanWork), uniGen.getRand()));
    pkt.sequenceNumber++;
    if( pkt.sequenceNumber == pkt.trainSize ) {// this->was the last packet
      pkt.sequenceNumber = 0;
      pkt.trainNumber++;
    }
    if( pkt.trainNumber == pkt.totalTrains ) {// this->was the last train
      trains[trainIndex] = PacketStruct(pairGen.getPair(), 
        expGen.getRand(meanTrainSize), expGen.getRand(meanTrainsPerComm), 
        expGen.getRand(meanWork), uniGen.getRand());
    }
    return packet;
  }
};
