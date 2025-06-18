import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'car_page.dart';
import 'profile_page.dart';
import 'my_car_page.dart';
import 'package:flutter_application_1/pages/service_center_page.dart';

class Car {
  final String model;
  final String efficiency;
  final String imageUrl;
  final String plate;
  final String tireDate;
  final int mileage;
  final String? docId;

  Car({
    required this.model,
    required this.efficiency,
    required this.imageUrl,
    required this.plate,
    required this.tireDate,
    required this.mileage,
    this.docId,
  });

  factory Car.fromMap(Map<String, dynamic> data, {String? docId}) {
    return Car(
      model: data['model'] ?? '',
      efficiency: data['efficiency'] ?? '',
      imageUrl: data['imageUrl']?.toString().trim() ?? '',
      plate: data['plate'] ?? '',
      tireDate: data['tireDate'] ?? '',
      mileage:
          data['mileage'] is int
              ? data['mileage']
              : int.tryParse(data['mileage']?.toString() ?? '0') ?? 0,
      docId: docId,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'model': model,
      'efficiency': efficiency,
      'imageUrl': imageUrl,
      'plate': plate,
      'tireDate': tireDate,
      'mileage': mileage,
    };
  }
}

class HomePage extends StatefulWidget {
  final String userName;

  const HomePage({required this.userName, Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  List<Car> _selectedCars = [];
  int _currentPage = 0;
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  void initState() {
    super.initState();
    _loadUserCars();
  }

  Future<void> _loadUserCars() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final carSnapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .get();

    setState(() {
      _selectedCars =
          carSnapshot.docs
              .map((doc) => Car.fromMap(doc.data(), docId: doc.id))
              .toList();
    });
  }

  Future<void> _addCar() async {
    final selectedCar = await Navigator.push<Car>(
      context,
      MaterialPageRoute(builder: (_) => CarPage()),
    );

    if (selectedCar != null) {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) return;

      final docRef = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('cars')
          .add(selectedCar.toMap());

      setState(() {
        _selectedCars.add(
          Car(
            model: selectedCar.model,
            efficiency: selectedCar.efficiency,
            imageUrl: selectedCar.imageUrl,
            plate: selectedCar.plate,
            tireDate: selectedCar.tireDate,
            mileage: selectedCar.mileage,
            docId: docRef.id,
          ),
        );
        _currentPage = _selectedCars.length - 1;
      });
    }
  }

  void _showCarOptions(int index) {
    showDialog(
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
                          Navigator.of(context).pop();

                          final user = FirebaseAuth.instance.currentUser;
                          if (user == null) return;

                          final carToRemove = _selectedCars[index];
                          if (carToRemove.docId != null) {
                            await FirebaseFirestore.instance
                                .collection('users')
                                .doc(user.uid)
                                .collection('cars')
                                .doc(carToRemove.docId)
                                .delete();
                          }

                          setState(() {
                            _selectedCars.removeAt(index);
                            if (_currentPage >= _selectedCars.length &&
                                _selectedCars.isNotEmpty) {
                              _currentPage = _selectedCars.length - 1;
                            }
                          });
                        },
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
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      endDrawer: _buildDrawer(),
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: Container(),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 20),
            child: IconButton(
              icon: Icon(Icons.menu, color: Colors.black),
              onPressed: () => _scaffoldKey.currentState?.openEndDrawer(),
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _buildCarSection(),
            SizedBox(height: 24),
            _buildDashboard(),
            SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: _iconButton(
                    '불량 진단',
                    Icons.camera_alt,
                    onTap: () {
                      // TODO: 진단 페이지 연결
                    },
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: _iconButton(
                    '정비소 찾기',
                    Icons.location_on,
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => ServiceCenterPage()),
                      );
                    },
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawer() {
    return Drawer(
      backgroundColor: Colors.white,
      child: Column(
        children: [
          Container(
            color: Colors.black87,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 40),
            width: double.infinity,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${widget.userName}님, 환영합니다.',
                  style: TextStyle(color: Colors.white, fontSize: 16),
                ),
                IconButton(
                  icon: Icon(Icons.close, color: Colors.white),
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ],
            ),
          ),
          ListTile(
            leading: Icon(Icons.person_outline),
            title: Text('프로필 수정'),
            onTap: () {
              Navigator.of(context).pop();
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => ProfileEditPage()),
              );
            },
          ),
          ListTile(
            leading: Icon(Icons.directions_car_outlined),
            title: Text('내 차 관리'),
            onTap: () {
              Navigator.of(context).pop();
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => MyCarPage()),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildCarSection() {
    return SizedBox(
      height: 240,
      child:
          _selectedCars.isEmpty
              ? GestureDetector(
                onTap: _addCar,
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.add_circle_outline,
                          size: 40,
                          color: Colors.grey,
                        ),
                        SizedBox(height: 8),
                        Text('차량 추가', style: TextStyle(color: Colors.grey)),
                      ],
                    ),
                  ),
                ),
              )
              : PageView.builder(
                itemCount: _selectedCars.length + 1,
                controller: PageController(
                  viewportFraction: 0.9,
                  initialPage: _currentPage,
                ),
                onPageChanged: (index) => setState(() => _currentPage = index),
                itemBuilder: (context, index) {
                  if (index == _selectedCars.length) {
                    return GestureDetector(
                      onTap: _addCar,
                      child: Container(
                        padding: EdgeInsets.symmetric(horizontal: 8),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade300,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Center(
                          child: Icon(
                            Icons.add_circle_outline,
                            size: 40,
                            color: Colors.grey,
                          ),
                        ),
                      ),
                    );
                  }

                  final car = _selectedCars[index];
                  return Container(
                    margin: EdgeInsets.symmetric(horizontal: 8),
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(color: Colors.black12, blurRadius: 4),
                      ],
                    ),
                    child: Stack(
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Center(
                              child: Image.network(
                                car.imageUrl,
                                height: 120,
                                fit: BoxFit.contain,
                                errorBuilder:
                                    (_, __, ___) =>
                                        Icon(Icons.image_not_supported),
                              ),
                            ),
                            SizedBox(height: 6),
                            Text(
                              car.model,
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            SizedBox(height: 2),
                            Text(
                              car.efficiency,
                              style: TextStyle(
                                color: Colors.black54,
                                fontSize: 12,
                              ),
                            ),
                            SizedBox(height: 2),
                            Text(
                              '타이어가 정상입니다.',
                              style: TextStyle(
                                color: Colors.redAccent,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                        Positioned(
                          top: 0,
                          right: 0,
                          child: IconButton(
                            icon: Icon(Icons.more_vert),
                            onPressed: () => _showCarOptions(index),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
    );
  }

  Widget _buildDashboard() {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('내 차량 대시보드', style: TextStyle(color: Colors.white)),
          SizedBox(height: 12),
          _dashboardItem('2024년 4월 20일', '> 120km'),
          SizedBox(height: 12),
          _dashboardItem('2025년 1월 3일', '> 2641km'),
        ],
      ),
    );
  }

  Widget _dashboardItem(String date, String distance) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              date,
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(distance, style: TextStyle(color: Colors.white60)),
          ],
        ),
        Icon(Icons.arrow_forward, color: Colors.white),
      ],
    );
  }

  Widget _iconButton(String label, IconData icon, {VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white),
            SizedBox(height: 8),
            Text(label, style: TextStyle(color: Colors.white)),
          ],
        ),
      ),
    );
  }
}
