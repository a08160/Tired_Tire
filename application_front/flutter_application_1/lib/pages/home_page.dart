import 'package:flutter/material.dart';
import 'diagnosis_menu_page.dart';

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF7F7F7),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 20),
            child: Icon(Icons.menu, color: Colors.black),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Container(
                width: double.infinity,
                height: 150,
                decoration: BoxDecoration(
                  color: Colors.grey.shade200,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.add_circle_outline,
                      size: 40,
                      color: Colors.grey,
                    ),
                    SizedBox(height: 8),
                    Text(
                      '차량 추가',
                      style: TextStyle(color: Colors.grey, fontSize: 16),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 24),
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '내 차량 대시보드',
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                    SizedBox(height: 12),
                    _dashboardItem('2024년 4월 20일', '> 120km'),
                    SizedBox(height: 12),
                    _dashboardItem('2025년 1월 3일', '> 2641km'),
                  ],
                ),
              ),
              SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: _iconButton(context, '불량 진단', Icons.camera_alt),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: _iconButton(context, '정비소 찾기', Icons.location_on),
                  ),
                ],
              ),
              SizedBox(height: 16),
              _iconButton(context, '게시판', Icons.article),
            ],
          ),
        ),
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
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 4),
            Text(distance, style: TextStyle(color: Colors.white60)),
          ],
        ),
        Icon(Icons.arrow_forward, color: Colors.white),
      ],
    );
  }

  Widget _iconButton(BuildContext context, String label, IconData icon) {
    return GestureDetector(
      onTap: () {
        if (label == '불량 진단') {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => DiagnosisMenuPage()),
          );
        }
      },
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 24),
            SizedBox(height: 8),
            Text(label, style: TextStyle(color: Colors.white)),
          ],
        ),
      ),
    );
  }
}
