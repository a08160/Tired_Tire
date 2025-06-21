import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:csv/csv.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Car {
  final String model;
  final String efficiency;
  final String imageUrl;

  Car({required this.model, required this.efficiency, required this.imageUrl});
}

enum WheelPosition { leftFront, leftRear, rightFront, rightRear }

class CarPage extends StatefulWidget {
  @override
  _CarPageState createState() => _CarPageState();
}

class _CarPageState extends State<CarPage> {
  List<Car> _cars = [];
  String _searchQuery = '';

  WheelPosition? _selectedWheel;
  Map<WheelPosition, String> _tireDates = {
    WheelPosition.leftFront: '',
    WheelPosition.leftRear: '',
    WheelPosition.rightFront: '',
    WheelPosition.rightRear: '',
  };

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
        if (row.length >= 3) {
          String rawUrl = row[2].toString();
          String imageUrl = rawUrl;
          try {
            final uri = Uri.parse(rawUrl);
            final srcParam = uri.queryParameters['src'];
            if (srcParam != null && srcParam.isNotEmpty) {
              imageUrl = Uri.decodeComponent(srcParam);
            }
          } catch (_) {}
          cars.add(
            Car(
              model: row[0].toString(),
              efficiency: row[1].toString(),
              imageUrl: imageUrl,
            ),
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

    // ✅ 여기에 추가
    final Map<WheelPosition, TextEditingController> tireControllers = {
      WheelPosition.leftFront: TextEditingController(),
      WheelPosition.leftRear: TextEditingController(),
      WheelPosition.rightFront: TextEditingController(),
      WheelPosition.rightRear: TextEditingController(),
    };
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

            child: StatefulBuilder(
              // ✅ 핵심 변경: StatefulBuilder 추가
              builder: (context, setState) {
                return SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '정보를 입력하세요.',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
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
                      Container(
                        height: 200,
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            Image.asset(
                              'assets/car_top_view.png',
                              fit: BoxFit.contain,
                            ),
                            Positioned(
                              top: 30,
                              left: 1,
                              child: GestureDetector(
                                onTap:
                                    () => setState(
                                      () =>
                                          _selectedWheel =
                                              WheelPosition.leftFront,
                                    ),
                                child: Image.asset(
                                  _selectedWheel == WheelPosition.leftFront
                                      ? 'assets/tire_blue.png'
                                      : 'assets/tire_black.png',
                                  width: 50,
                                ),
                              ),
                            ),
                            Positioned(
                              bottom: 30,
                              left: 1,
                              child: GestureDetector(
                                onTap:
                                    () => setState(
                                      () =>
                                          _selectedWheel =
                                              WheelPosition.leftRear,
                                    ),
                                child: Image.asset(
                                  _selectedWheel == WheelPosition.leftRear
                                      ? 'assets/tire_blue.png'
                                      : 'assets/tire_black.png',
                                  width: 50,
                                ),
                              ),
                            ),
                            Positioned(
                              top: 30,
                              right: 1,
                              child: GestureDetector(
                                onTap:
                                    () => setState(
                                      () =>
                                          _selectedWheel =
                                              WheelPosition.rightFront,
                                    ),
                                child: Image.asset(
                                  _selectedWheel == WheelPosition.rightFront
                                      ? 'assets/tire_blue.png'
                                      : 'assets/tire_black.png',
                                  width: 50,
                                ),
                              ),
                            ),
                            Positioned(
                              bottom: 30,
                              right: 1,
                              child: GestureDetector(
                                onTap:
                                    () => setState(
                                      () =>
                                          _selectedWheel =
                                              WheelPosition.rightRear,
                                    ),
                                child: Image.asset(
                                  _selectedWheel == WheelPosition.rightRear
                                      ? 'assets/tire_blue.png'
                                      : 'assets/tire_black.png',
                                  width: 50,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(height: 16),
                      if (_selectedWheel != null)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '선택된 바퀴: ${_selectedWheel.toString().split('.').last}',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            SizedBox(height: 8),
                            InkWell(
                              onTap: () async {
                                DateTime? picked = await showDatePicker(
                                  context: context,
                                  initialDate: DateTime.now(),
                                  firstDate: DateTime(2015),
                                  lastDate: DateTime.now(),
                                );
                                if (picked != null) {
                                  String formatted =
                                      "${picked.year}-${picked.month.toString().padLeft(2, '0')}-${picked.day.toString().padLeft(2, '0')}";
                                  setState(() {
                                    tireControllers[_selectedWheel!]!.text =
                                        formatted;
                                    _tireDates[_selectedWheel!] = formatted;
                                  });
                                }
                              },
                              child: IgnorePointer(
                                child: TextField(
                                  controller: tireControllers[_selectedWheel],
                                  decoration: _inputDecoration(
                                    '제조일 선택 (예: 2023-06-19)',
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
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

                                if (plate.isEmpty ||
                                    mileage.isEmpty ||
                                    int.tryParse(mileage) == null) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                      content: Text('차량번호와 주행거리를 정확히 입력해주세요.'),
                                    ),
                                  );
                                  return;
                                }

                                final incompletePositions =
                                    _tireDates.entries
                                        .where(
                                          (entry) => entry.value.trim().isEmpty,
                                        )
                                        .map(
                                          (entry) =>
                                              entry.key
                                                  .toString()
                                                  .split('.')
                                                  .last,
                                        )
                                        .toList();

                                if (incompletePositions.isNotEmpty) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                      content: Text(
                                        '다음 위치의 타이어 제조일을 입력해주세요: ${incompletePositions.join(', ')}',
                                      ),
                                    ),
                                  );
                                  return;
                                }

                                final user = FirebaseAuth.instance.currentUser;
                                if (user == null) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text('로그인이 필요합니다.')),
                                  );
                                  return;
                                }

                                final userId = user.uid;

                                final carData = {
                                  'model': car.model,
                                  'efficiency': car.efficiency,
                                  'imageUrl': car.imageUrl,
                                  'plate': plate,
                                  'mileage': int.parse(mileage),
                                  'createdAt': FieldValue.serverTimestamp(),
                                  'tireDateLeftFront':
                                      _tireDates[WheelPosition.leftFront],
                                  'tireDateLeftRear':
                                      _tireDates[WheelPosition.leftRear],
                                  'tireDateRightFront':
                                      _tireDates[WheelPosition.rightFront],
                                  'tireDateRightRear':
                                      _tireDates[WheelPosition.rightRear],
                                };

                                try {
                                  await FirebaseFirestore.instance
                                      .collection('users')
                                      .doc(userId)
                                      .collection('cars')
                                      .add(carData);

                                  if (!mounted) return;
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                      content: Text('차량이 성공적으로 등록되었습니다.'),
                                    ),
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
                );
              },
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
      backgroundColor: Colors.white, // 전체 배경 완전 흰색으로 변경
      appBar: AppBar(
        backgroundColor: Colors.transparent, // AppBar 배경 투명
        elevation: 0, // 그림자 제거
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
                                    child: Image.network(
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
