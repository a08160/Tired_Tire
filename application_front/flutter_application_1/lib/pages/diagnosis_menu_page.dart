import 'package:flutter/material.dart';
import 'image_picker_page.dart';

class DiagnosisMenuPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A171D),
      appBar: AppBar(
        title: Text('불량 진단 메뉴', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                // TODO: 타이어 공기압 진단 기능으로 이동
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => ImagePickerPage()),
                );
              },
              style: ElevatedButton.styleFrom(
                minimumSize: Size(double.infinity, 60),
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              child: Text('기능 1: 타이어 공기압 진단'),
            ),
            SizedBox(height: 24),
            ElevatedButton(
              onPressed: () {
                // TODO: 타이어 균열 진단 기능으로 이동
              },
              style: ElevatedButton.styleFrom(
                minimumSize: Size(double.infinity, 60),
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              child: Text('기능 2: 타이어 균열 진단'),
            ),
          ],
        ),
      ),
    );
  }
}
