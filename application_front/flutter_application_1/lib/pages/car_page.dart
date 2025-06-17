import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:csv/csv.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class Car {
  final String model;
  final String efficiency;
  final String imageUrl;

  Car({required this.model, required this.efficiency, required this.imageUrl});
}

class CarPage extends StatefulWidget {
  @override
  _CarPageState createState() => _CarPageState();
}

class _CarPageState extends State<CarPage> {
  List<Car> _cars = [];
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    _loadCSV();
  }

  void _loadCSV() async {
    try {
      final rawData = await rootBundle.loadString("assets/car_data.csv");
      List<List<dynamic>> listData = const CsvToListConverter().convert(
        rawData,
        eol: '\n',
      );

      List<Car> cars = [];
      for (int i = 1; i < listData.length; i++) {
        final row = listData[i];
        if (row.length >= 2) {
          final model = row[0].toString();
          final efficiency = row[1].toString();

          final imageFileName = model; // 공백 그대로 유지
          final assetImagePath = 'assets/car_images/$imageFileName.jpg';

          cars.add(
            Car(model: model, efficiency: efficiency, imageUrl: assetImagePath),
          );
        }
      }

      if (!mounted) return;
      setState(() => _cars = cars);
    } catch (e) {
      print('CSV 로드 중 오류: $e');
    }
  }

  void _showRegisterDialog(Car car) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Container(
            width: 240,
            padding: EdgeInsets.symmetric(vertical: 20, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '차량을 등록하시겠습니까?',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () => Navigator.of(context).pop(),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color(0xFFE8E8E8),
                          foregroundColor: Color(0xFF666666),
                        ),
                        child: Text('취소'),
                      ),
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).pop();
                          _showInputPlateDialog(car);
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color(0xFF282931),
                        ),
                        child: Text(
                          '등록',
                          style: TextStyle(color: Colors.white),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  void _showInputPlateDialog(Car car) {
    final plateController = TextEditingController();
    final mileageController = TextEditingController();
    final tireDateController = TextEditingController();

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Container(
            width: 240,
            padding: EdgeInsets.symmetric(vertical: 20, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
            ),
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '정보를 입력하세요.',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 16),
                  TextField(
                    controller: plateController,
                    cursorColor: Colors.black,
                    decoration: _inputDecoration('차량번호 입력'),
                  ),
                  SizedBox(height: 12),
                  TextField(
                    controller: mileageController,
                    cursorColor: Colors.black,
                    keyboardType: TextInputType.number,
                    decoration: _inputDecoration('주행거리 입력 (km)'),
                  ),
                  SizedBox(height: 12),
                  TextField(
                    controller: tireDateController,
                    cursorColor: Colors.black,
                    decoration: _inputDecoration('타이어 제조일자 입력 (예: 2023-05-01)'),
                  ),
                  SizedBox(height: 20),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () => Navigator.of(context).pop(),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFFE8E8E8),
                            foregroundColor: Color(0xFF666666),
                          ),
                          child: Text('취소'),
                        ),
                      ),
                      SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () async {
                            final plate = plateController.text.trim();
                            final mileage = mileageController.text.trim();
                            final tireDate = tireDateController.text.trim();

                            if (plate.isEmpty ||
                                mileage.isEmpty ||
                                tireDate.isEmpty ||
                                int.tryParse(mileage) == null) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('입력값을 다시 확인해주세요.')),
                              );
                              return;
                            }

                            final userId = 't9bvNqNveIQ5xtiTkrfxOJRHbEY2';
                            final carData = {
                              'model': car.model,
                              'efficiency': car.efficiency,
                              'imageUrl': car.imageUrl,
                              'plate': plate,
                              'mileage': int.parse(mileage),
                              'tireDate': tireDate,
                              'createdAt': FieldValue.serverTimestamp(),
                            };

                            try {
                              await FirebaseFirestore.instance
                                  .collection('users')
                                  .doc(userId)
                                  .collection('cars')
                                  .add(carData);

                              if (!mounted) return;
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('차량이 성공적으로 등록되었습니다.')),
                              );

                              Navigator.of(context).pop(carData);
                            } catch (e) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('등록 실패: $e')),
                              );
                            }
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF282931),
                          ),
                          child: Text(
                            '등록',
                            style: TextStyle(color: Colors.white),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  InputDecoration _inputDecoration(String label) {
    return InputDecoration(
      labelText: label,
      labelStyle: TextStyle(color: Colors.grey),
      floatingLabelStyle: TextStyle(color: Colors.black),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
      enabledBorder: OutlineInputBorder(
        borderSide: BorderSide(color: Colors.black),
      ),
      focusedBorder: OutlineInputBorder(
        borderSide: BorderSide(color: Colors.black),
      ),
      contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    );
  }

  @override
  Widget build(BuildContext context) {
    final filteredCars =
        _cars
            .where(
              (car) =>
                  car.model.toLowerCase().contains(_searchQuery.toLowerCase()),
            )
            .toList();

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black87),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 20, 16, 10),
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.black87),
                  borderRadius: BorderRadius.circular(24),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Row(
                  children: [
                    Icon(Icons.search, color: Colors.black54),
                    SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        cursorColor: Colors.black,
                        onChanged:
                            (value) => setState(() => _searchQuery = value),
                        decoration: InputDecoration(
                          hintText: 'Search',
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                    if (_searchQuery.isNotEmpty)
                      GestureDetector(
                        onTap: () => setState(() => _searchQuery = ''),
                        child: Icon(Icons.close, color: Colors.black54),
                      ),
                  ],
                ),
              ),
            ),
            Expanded(
              child:
                  filteredCars.isEmpty
                      ? Center(child: Text('검색 결과가 없습니다.'))
                      : ListView.builder(
                        padding: EdgeInsets.symmetric(horizontal: 20),
                        itemCount: filteredCars.length,
                        itemBuilder: (context, index) {
                          final car = filteredCars[index];
                          return GestureDetector(
                            onTap: () => _showRegisterDialog(car),
                            child: Padding(
                              padding: const EdgeInsets.only(bottom: 28),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  AspectRatio(
                                    aspectRatio: 16 / 9,
                                    child: Image.asset(
                                      car.imageUrl,
                                      fit: BoxFit.contain,
                                      errorBuilder:
                                          (context, error, stackTrace) =>
                                              Container(
                                                color: Colors.grey[200],
                                                child: Icon(
                                                  Icons.car_repair,
                                                  size: 64,
                                                ),
                                              ),
                                    ),
                                  ),
                                  SizedBox(height: 10),
                                  Text(
                                    car.model,
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Text(
                                    '${car.efficiency} km/L',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.black54,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
            ),
          ],
        ),
      ),
    );
  }
}
