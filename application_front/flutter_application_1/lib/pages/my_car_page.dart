import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Car {
  final String? docId;
  final String plate;
  final String model;
  final int mileage;
  final String tireDate;

  Car({
    this.docId,
    required this.plate,
    required this.model,
    required this.mileage,
    required this.tireDate,
  });

  factory Car.fromMap(Map<String, dynamic> data, String docId) {
    return Car(
      docId: docId,
      plate: data['plate'] ?? '',
      model: data['model'] ?? '',
      mileage: data['mileage'] ?? 0,
      tireDate: data['tireDate'] ?? '',
    );
  }
}

class MyCarPage extends StatefulWidget {
  const MyCarPage({Key? key}) : super(key: key);

  @override
  State<MyCarPage> createState() => _MyCarPageState();
}

class _MyCarPageState extends State<MyCarPage> {
  List<Car> cars = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchCars();
  }

  Future<void> _fetchCars() async {
    setState(() {
      isLoading = true;
    });

    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() {
        cars = [];
        isLoading = false;
      });
      return;
    }

    final snapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .get();

    final loadedCars =
        snapshot.docs.map((doc) => Car.fromMap(doc.data(), doc.id)).toList();

    setState(() {
      cars = loadedCars;
      isLoading = false;
    });
  }

  void _showEditDialog(BuildContext context, Car car) {
    final plateController = TextEditingController(text: car.plate);
    final mileageController = TextEditingController(
      text: car.mileage.toString(),
    );
    final tireDateController = TextEditingController(text: car.tireDate);

    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Container(
            width: 280,
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
                    '차량 정보 수정',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 16),
                  TextField(
                    controller: plateController,
                    cursorColor: Colors.black,
                    decoration: _inputDecoration('차량 번호 입력'),
                  ),
                  SizedBox(height: 12),
                  TextField(
                    controller: mileageController,
                    cursorColor: Colors.black,
                    keyboardType: TextInputType.number,
                    decoration: _inputDecoration('주행 거리 입력 (km)'),
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

                            final user = FirebaseAuth.instance.currentUser;
                            if (user == null || car.docId == null) return;

                            final updatedData = {
                              'plate': plate,
                              'mileage': int.parse(mileage),
                              'tireDate': tireDate,
                            };

                            try {
                              await FirebaseFirestore.instance
                                  .collection('users')
                                  .doc(user.uid)
                                  .collection('cars')
                                  .doc(car.docId)
                                  .update(updatedData);

                              if (!mounted) return;

                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('차량 정보가 수정되었습니다.')),
                              );

                              Navigator.of(context).pop();
                              _fetchCars();
                            } catch (e) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('수정 실패: $e')),
                              );
                            }
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF282931),
                          ),
                          child: Text(
                            '저장',
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

  void _deleteCar(Car car) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          content: SizedBox(
            width: 240,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '삭제하시겠습니까?',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                ),
                SizedBox(height: 8),
                Text('선택한 차량 정보를 삭제합니다.'),
                SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: () => Navigator.of(context).pop(false),
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
                        onPressed: () => Navigator.of(context).pop(true),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.black,
                        ),
                        child: Text(
                          '삭제',
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

    if (confirmed == true) {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null || car.docId == null) return;

      await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('cars')
          .doc(car.docId!)
          .delete();

      _fetchCars();
    }
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
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        centerTitle: true,
        title: Text('내 차 관리', style: TextStyle(color: Color(0xFF282931))),
        elevation: 0,
        iconTheme: IconThemeData(color: Colors.grey),
      ),
      body:
          isLoading
              ? Center(child: CircularProgressIndicator())
              : cars.isEmpty
              ? Center(
                child: Text('등록된 차량이 없습니다.', style: TextStyle(fontSize: 18)),
              )
              : ListView.builder(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                itemCount: cars.length,
                itemBuilder: (context, index) {
                  final car = cars[index];
                  return Container(
                    margin: EdgeInsets.symmetric(vertical: 10),
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Color(0xFFF6F6F6),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  car.model,
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                SizedBox(height: 4),
                                Text(car.plate, style: TextStyle(fontSize: 16)),
                              ],
                            ),
                            Image.asset(
                              'assets/car_images/${car.model}.jpg',
                              width: 120,
                            ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              '타이어: ${car.tireDate}',
                              style: TextStyle(color: Colors.grey),
                            ),
                            Text(
                              '${car.mileage} km',
                              style: TextStyle(color: Colors.grey),
                            ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            OutlinedButton(
                              onPressed: () => _deleteCar(car),
                              style: OutlinedButton.styleFrom(
                                side: BorderSide(color: Colors.black),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                padding: EdgeInsets.symmetric(
                                  horizontal: 24,
                                  vertical: 10,
                                ),
                              ),
                              child: Text(
                                '삭제',
                                style: TextStyle(color: Colors.black),
                              ),
                            ),
                            ElevatedButton(
                              onPressed: () => _showEditDialog(context, car),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Color(0xFF282931),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                padding: EdgeInsets.symmetric(
                                  horizontal: 24,
                                  vertical: 10,
                                ),
                              ),
                              child: Text(
                                '수정',
                                style: TextStyle(color: Colors.white),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  );
                },
              ),
    );
  }
}
