import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Car {
  final String? docId;
  final String plate;
  final String model;
  final int mileage;
  final String? imageUrl;
  final String tireDateLeftFront;
  final String tireDateLeftRear;
  final String tireDateRightFront;
  final String tireDateRightRear;

  Car({
    this.docId,
    required this.plate,
    required this.model,
    required this.mileage,
    required this.imageUrl,
    required this.tireDateLeftFront,
    required this.tireDateLeftRear,
    required this.tireDateRightFront,
    required this.tireDateRightRear,
  });

  factory Car.fromMap(Map<String, dynamic> data, String docId) {
    return Car(
      docId: docId,
      plate: data['plate'] ?? '',
      model: data['model'] ?? '',
      mileage:
          data['mileage'] is int
              ? data['mileage']
              : int.tryParse(data['mileage']?.toString() ?? '0') ?? 0,
      imageUrl:
          data.containsKey('imageUrl')
              ? data['imageUrl']?.toString() ?? ''
              : '',
      tireDateLeftFront: data['tireDateLeftFront'] ?? '',
      tireDateLeftRear: data['tireDateLeftRear'] ?? '',
      tireDateRightFront: data['tireDateRightFront'] ?? '',
      tireDateRightRear: data['tireDateRightRear'] ?? '',
    );
  }
}

class MyCarPage extends StatefulWidget {
  const MyCarPage({Key? key}) : super(key: key);

  @override
  State<MyCarPage> createState() => _MyCarPageState();
}

enum WheelPosition { leftFront, leftRear, rightFront, rightRear }

class _MyCarPageState extends State<MyCarPage> {
  List<Car> cars = [];
  bool isLoading = true;

  WheelPosition? _selectedWheel;

  final Map<WheelPosition, String> _tireDates = {
    WheelPosition.leftFront: '',
    WheelPosition.leftRear: '',
    WheelPosition.rightFront: '',
    WheelPosition.rightRear: '',
  };

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

    try {
      final snapshot =
          await FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .collection('cars')
              .get();

      final loadedCars =
          snapshot.docs
              .map((doc) {
                try {
                  return Car.fromMap(doc.data(), doc.id);
                } catch (e) {
                  print('[🚨 Car.fromMap 오류] 문서: ${doc.id}, 에러: $e');
                  return null;
                }
              })
              .whereType<Car>()
              .toList();

      setState(() {
        cars = loadedCars;
        isLoading = false;
      });
    } catch (e) {
      print('[🚨 Firestore 전체 실패] 에러: $e');
      setState(() {
        isLoading = false;
      });
    }
  }

  void _showEditDialog(BuildContext context, Car car) {
    final plateController = TextEditingController(text: car.plate);
    final mileageController = TextEditingController(
      text: car.mileage.toString(),
    );

    final Map<WheelPosition, TextEditingController> tireControllers = {
      WheelPosition.leftFront: TextEditingController(
        text: car.tireDateLeftFront ?? '',
      ),
      WheelPosition.leftRear: TextEditingController(
        text: car.tireDateLeftRear ?? '',
      ),
      WheelPosition.rightFront: TextEditingController(
        text: car.tireDateRightFront ?? '',
      ),
      WheelPosition.rightRear: TextEditingController(
        text: car.tireDateRightRear ?? '',
      ),
    };

    WheelPosition? selectedWheel = WheelPosition.leftFront; // 초기 선택

    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Container(
            padding: EdgeInsets.all(20),
            width: 280,
            child: StatefulBuilder(
              builder: (context, setState) {
                return SingleChildScrollView(
                  child: Column(
                    children: [
                      Text(
                        '차량 정보 수정',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      SizedBox(height: 16),
                      TextField(
                        controller: plateController,
                        decoration: _inputDecoration('차량 번호 입력'),
                      ),
                      SizedBox(height: 12),
                      TextField(
                        controller: mileageController,
                        keyboardType: TextInputType.number,
                        decoration: _inputDecoration('주행 거리 입력 (km)'),
                      ),
                      SizedBox(height: 12),
                      Container(
                        height: 200,
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            Image.asset('assets/car_top_view.png'),
                            for (final pos in WheelPosition.values)
                              Positioned(
                                top:
                                    (pos == WheelPosition.leftFront ||
                                            pos == WheelPosition.rightFront)
                                        ? 30
                                        : null,
                                bottom:
                                    (pos == WheelPosition.leftRear ||
                                            pos == WheelPosition.rightRear)
                                        ? 30
                                        : null,
                                left:
                                    (pos == WheelPosition.leftFront ||
                                            pos == WheelPosition.leftRear)
                                        ? 10
                                        : null,
                                right:
                                    (pos == WheelPosition.rightFront ||
                                            pos == WheelPosition.rightRear)
                                        ? 10
                                        : null,
                                child: GestureDetector(
                                  onTap:
                                      () => setState(() => selectedWheel = pos),
                                  child: Image.asset(
                                    selectedWheel == pos
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
                      Text(
                        '선택된 바퀴: ${selectedWheel.toString().split('.').last}',
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
                              if (selectedWheel == null) return;
                              tireControllers[selectedWheel]!.text = formatted;
                            });
                          }
                        },
                        child: IgnorePointer(
                          child: TextField(
                            controller: tireControllers[selectedWheel],
                            decoration: _inputDecoration(
                              '제조일 선택 (예: 2023-06-19)',
                            ),
                          ),
                        ),
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

                                if (plate.isEmpty ||
                                    mileage.isEmpty ||
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
                                  'tireDateLeftFront':
                                      tireControllers[WheelPosition.leftFront]!
                                          .text,
                                  'tireDateLeftRear':
                                      tireControllers[WheelPosition.leftRear]!
                                          .text,
                                  'tireDateRightFront':
                                      tireControllers[WheelPosition.rightFront]!
                                          .text,
                                  'tireDateRightRear':
                                      tireControllers[WheelPosition.rightRear]!
                                          .text,
                                };

                                try {
                                  await FirebaseFirestore.instance
                                      .collection('users')
                                      .doc(user.uid)
                                      .collection('cars')
                                      .doc(car.docId)
                                      .update(updatedData);

                                  if (!mounted) return;
                                  Navigator.of(context).pop();
                                  _fetchCars();
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(content: Text('차량 정보가 수정되었습니다.')),
                                  );
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
                                '수정',
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
                            (car.imageUrl ?? '').isNotEmpty
                                ? Image.network(
                                  car.imageUrl!,
                                  width: 200,
                                  fit: BoxFit.contain,
                                )
                                : Image.asset(
                                  'assets/car_placeholder.png',
                                  width: 120,
                                ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
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
